from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


class SkillValidationError(ValueError):
    """Raised when a Skill package does not satisfy the declarative contract."""


@dataclass(frozen=True)
class SkillTargets:
    agent_roles: Tuple[str, ...]
    intent_categories: Tuple[str, ...]
    goals: Tuple[str, ...]
    tools: Tuple[str, ...]


@dataclass(frozen=True)
class SkillDefinition:
    id: str
    name: str
    version: str
    description: str
    targets: SkillTargets
    allowed_tools: Tuple[str, ...]
    prompt: str
    priority: int
    source: str


@dataclass(frozen=True)
class SkillSnapshot:
    generation: int
    skills: Mapping[str, SkillDefinition]


@dataclass(frozen=True)
class SkillUpload:
    draft_id: str
    status: str
    auto_published: bool
    skill: SkillDefinition


class SkillRuntime:
    """Loads validated Skill directories and exposes stable request snapshots."""

    _ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
    _VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
    _VALID_ROLES = {"general", "technical", "billing", "escalation"}
    _VALID_GOALS = {"knowledge", "live_record", "action"}
    _MAX_MANIFEST_BYTES = 64 * 1024
    _MAX_PROMPT_BYTES = 32 * 1024
    _MAX_REFRESH_ATTEMPTS = 3

    def __init__(
        self,
        *,
        catalog_dir: Path | str,
        published_dir: Path | str,
        drafts_dir: Path | str,
        known_tools: Iterable[str],
    ):
        self.catalog_dir = Path(catalog_dir)
        self.published_dir = Path(published_dir)
        self.drafts_dir = Path(drafts_dir)
        self._known_tools = frozenset(str(tool) for tool in known_tools)
        self._lock = threading.Lock()
        self._snapshot = SkillSnapshot(generation=0, skills=MappingProxyType({}))
        self._fingerprint: Tuple[Tuple[str, int, int], ...] | None = None
        self._last_refresh_at: str | None = None
        self.last_errors: Tuple[str, ...] = ()

    @property
    def snapshot(self) -> SkillSnapshot:
        return self._snapshot

    def refresh(self) -> SkillSnapshot:
        """Atomically replace the active catalog only after full validation."""
        for _ in range(self._MAX_REFRESH_ATTEMPTS):
            try:
                before = self._source_fingerprint()
                loaded = self._load_active_skills()
                after = self._source_fingerprint()
            except (OSError, SkillValidationError) as exc:
                self.last_errors = (str(exc),)
                return self._snapshot
            if before != after:
                continue

            with self._lock:
                self._snapshot = SkillSnapshot(
                    generation=self._snapshot.generation + 1,
                    skills=MappingProxyType(loaded),
                )
                self._fingerprint = after
                self._last_refresh_at = datetime.now(timezone.utc).isoformat()
                self.last_errors = ()
                return self._snapshot

        self.last_errors = ("Skill 源在刷新期间持续变化",)
        return self._snapshot

    def refresh_if_changed(self) -> bool:
        if self._fingerprint == self._source_fingerprint():
            return False
        before = self._snapshot.generation
        self.refresh()
        return self._snapshot.generation != before

    def describe(self) -> Dict[str, object]:
        return {
            "generation": self._snapshot.generation,
            "last_refresh_at": self._last_refresh_at,
            "active": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "version": skill.version,
                    "source": skill.source,
                    "priority": skill.priority,
                    "allowed_tools": list(skill.allowed_tools),
                }
                for skill in sorted(self._snapshot.skills.values(), key=lambda item: item.id)
            ],
            "last_errors": list(self.last_errors),
        }

    def select(
        self,
        *,
        agent_role: str,
        intent_category: str,
        goal: str,
        planned_tools: Sequence[str],
    ) -> List[SkillDefinition]:
        return self.select_from_snapshot(
            self._snapshot,
            agent_role=agent_role,
            intent_category=intent_category,
            goal=goal,
            planned_tools=planned_tools,
        )

    def select_from_snapshot(
        self,
        snapshot: SkillSnapshot,
        *,
        agent_role: str,
        intent_category: str,
        goal: str,
        planned_tools: Sequence[str],
    ) -> List[SkillDefinition]:
        tool_set = set(planned_tools)
        selected = [
            skill for skill in snapshot.skills.values()
            if self._matches(skill, agent_role, intent_category, goal, tool_set)
        ]
        return sorted(selected, key=lambda skill: (-skill.priority, skill.id))

    def _load_active_skills(self) -> Dict[str, SkillDefinition]:
        candidates: List[SkillDefinition] = []
        candidates.extend(self._load_source(self.catalog_dir, source="catalog"))
        candidates.extend(self._load_source(self.published_dir, source="published"))

        selected: Dict[str, SkillDefinition] = {}
        versions: set[Tuple[str, str]] = set()
        for skill in candidates:
            identity = (skill.id, skill.version)
            if identity in versions:
                raise SkillValidationError(f"Skill 版本重复: {skill.id}@{skill.version}")
            versions.add(identity)
            current = selected.get(skill.id)
            if current is None or self._prefer(skill, current):
                selected[skill.id] = skill
        return selected

    def _load_source(self, root: Path, *, source: str) -> List[SkillDefinition]:
        if not root.exists():
            return []
        packages = sorted(root.glob("*/*/skill.json"))
        return [self._parse_package(manifest_path, source=source) for manifest_path in packages]

    def _source_fingerprint(self) -> Tuple[Tuple[str, int, int], ...]:
        entries: List[Tuple[str, int, int]] = []
        for root in (self.catalog_dir, self.published_dir):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                stat = path.stat()
                entries.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
        return tuple(sorted(entries))

    def _parse_package(self, manifest_path: Path, *, source: str) -> SkillDefinition:
        if manifest_path.stat().st_size > self._MAX_MANIFEST_BYTES:
            raise SkillValidationError(f"Skill 清单过大: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillValidationError(f"Skill 清单无效: {manifest_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SkillValidationError(f"Skill 清单必须是对象: {manifest_path}")

        skill_id = self._required_text(payload, "id", manifest_path)
        if not self._ID_PATTERN.fullmatch(skill_id):
            raise SkillValidationError(f"Skill id 非法: {skill_id}")
        version = self._required_text(payload, "version", manifest_path)
        if not self._VERSION_PATTERN.fullmatch(version):
            raise SkillValidationError(f"Skill version 非法: {version}")

        targets_payload = payload.get("targets")
        if not isinstance(targets_payload, dict):
            raise SkillValidationError(f"Skill targets 缺失或非法: {skill_id}")
        targets = SkillTargets(
            agent_roles=self._string_list(targets_payload, "agent_roles", skill_id),
            intent_categories=self._string_list(targets_payload, "intent_categories", skill_id),
            goals=self._string_list(targets_payload, "goals", skill_id),
            tools=self._string_list(targets_payload, "tools", skill_id, required=False),
        )
        unknown_roles = set(targets.agent_roles) - self._VALID_ROLES
        if unknown_roles:
            raise SkillValidationError(f"Skill {skill_id} 包含未知 Agent: {sorted(unknown_roles)}")
        unknown_goals = set(targets.goals) - self._VALID_GOALS
        if unknown_goals:
            raise SkillValidationError(f"Skill {skill_id} 包含未知工作流目标: {sorted(unknown_goals)}")

        allowed_tools = self._string_list(payload, "allowed_tools", skill_id, required=False)
        unknown_tools = (set(targets.tools) | set(allowed_tools)) - self._known_tools
        if unknown_tools:
            raise SkillValidationError(f"Skill {skill_id} 绑定了未知工具: {sorted(unknown_tools)}")

        prompt_file = self._required_text(payload, "prompt_file", manifest_path)
        if Path(prompt_file).name != prompt_file:
            raise SkillValidationError(f"Skill {skill_id} 的 prompt_file 不能包含目录")
        prompt_path = manifest_path.parent / prompt_file
        if not prompt_path.is_file() or prompt_path.stat().st_size > self._MAX_PROMPT_BYTES:
            raise SkillValidationError(f"Skill {skill_id} 的提示词文件缺失或过大")
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise SkillValidationError(f"Skill {skill_id} 的提示词文件不是 UTF-8") from exc
        if not prompt:
            raise SkillValidationError(f"Skill {skill_id} 的提示词不能为空")

        priority = payload.get("priority", 0)
        if not isinstance(priority, int) or isinstance(priority, bool) or not -1000 <= priority <= 1000:
            raise SkillValidationError(f"Skill {skill_id} 的 priority 非法")
        return SkillDefinition(
            id=skill_id,
            name=self._required_text(payload, "name", manifest_path),
            version=version,
            description=self._required_text(payload, "description", manifest_path),
            targets=targets,
            allowed_tools=allowed_tools,
            prompt=prompt,
            priority=priority,
            source=source,
        )

    @staticmethod
    def _required_text(payload: Mapping[str, object], key: str, location: Path) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SkillValidationError(f"Skill 字段 {key} 缺失或非法: {location}")
        return value.strip()

    @staticmethod
    def _string_list(
        payload: Mapping[str, object], key: str, skill_id: str, *, required: bool = True
    ) -> Tuple[str, ...]:
        value = payload.get(key, [] if not required else None)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise SkillValidationError(f"Skill {skill_id} 的 {key} 必须是字符串数组")
        return tuple(dict.fromkeys(item.strip() for item in value))

    @staticmethod
    def _version_key(version: str) -> Tuple[int, int, int]:
        return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]

    def _prefer(self, candidate: SkillDefinition, current: SkillDefinition) -> bool:
        source_rank = {"catalog": 0, "published": 1}
        candidate_rank = source_rank[candidate.source]
        current_rank = source_rank[current.source]
        return (candidate_rank, self._version_key(candidate.version)) > (
            current_rank,
            self._version_key(current.version),
        )

    @staticmethod
    def _matches(
        skill: SkillDefinition,
        agent_role: str,
        intent_category: str,
        goal: str,
        planned_tools: set[str],
    ) -> bool:
        return (
            agent_role in skill.targets.agent_roles
            and intent_category in skill.targets.intent_categories
            and goal in skill.targets.goals
            and set(skill.targets.tools).issubset(planned_tools)
            and set(skill.allowed_tools).issubset(planned_tools)
        )


class SkillStore:
    """Persists validated uploaded packages and promotes reviewed versions."""

    _PACKAGE_FILES = {"skill.json", "prompt.md"}
    _MAX_ARCHIVE_BYTES = 128 * 1024

    def __init__(self, *, runtime: SkillRuntime, review_required: bool):
        self._runtime = runtime
        self._review_required = review_required

    def upload_zip(self, archive_bytes: bytes) -> SkillUpload:
        package_files = self._read_package_files(archive_bytes)
        draft_id = uuid.uuid4().hex
        staging_dir = self._runtime.drafts_dir / f".staging-{draft_id}"
        package_dir = staging_dir / "package"
        package_dir.mkdir(parents=True, exist_ok=False)
        for name, content in package_files.items():
            (package_dir / name).write_bytes(content)

        skill = self._runtime._parse_package(package_dir / "skill.json", source="draft")
        final_dir = self._runtime.drafts_dir / draft_id / skill.id / skill.version
        final_dir.parent.mkdir(parents=True, exist_ok=False)
        os.replace(package_dir, final_dir)

        if self._review_required:
            return SkillUpload(draft_id=draft_id, status="draft", auto_published=False, skill=skill)

        self.publish(draft_id)
        return SkillUpload(draft_id=draft_id, status="published", auto_published=True, skill=skill)

    def publish(self, draft_id: str) -> SkillDefinition:
        manifest_paths = list((self._runtime.drafts_dir / draft_id).glob("*/*/skill.json"))
        if len(manifest_paths) != 1:
            raise SkillValidationError(f"Skill 草稿不存在或结构非法: {draft_id}")

        manifest_path = manifest_paths[0]
        skill = self._runtime._parse_package(manifest_path, source="published")
        target_dir = self._runtime.published_dir / skill.id / skill.version
        if target_dir.exists():
            raise SkillValidationError(f"Skill 版本已发布: {skill.id}@{skill.version}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        staging_dir = self._runtime.published_dir / f".staging-{uuid.uuid4().hex}"
        staged_package = staging_dir / skill.id / skill.version
        staged_package.mkdir(parents=True, exist_ok=False)
        for name in self._PACKAGE_FILES:
            (staged_package / name).write_bytes((manifest_path.parent / name).read_bytes())
        os.replace(staged_package, target_dir)
        snapshot = self._runtime.refresh()
        active = snapshot.skills.get(skill.id)
        if active is None or active.version != skill.version or active.source != "published":
            shutil.rmtree(target_dir)
            errors = "; ".join(self._runtime.last_errors) or "Skill 刷新未激活"
            raise SkillValidationError(f"Skill 发布未生效: {errors}")
        return skill

    def list_drafts(self) -> List[Dict[str, str]]:
        if not self._runtime.drafts_dir.exists():
            return []
        drafts: List[Dict[str, str]] = []
        for draft_dir in sorted(self._runtime.drafts_dir.iterdir()):
            if not draft_dir.is_dir() or draft_dir.name.startswith("."):
                continue
            manifest_paths = list(draft_dir.glob("*/*/skill.json"))
            if len(manifest_paths) != 1:
                drafts.append({"draft_id": draft_dir.name, "error": "草稿结构非法"})
                continue
            try:
                skill = self._runtime._parse_package(manifest_paths[0], source="draft")
            except SkillValidationError as exc:
                drafts.append({"draft_id": draft_dir.name, "error": str(exc)})
                continue
            drafts.append({
                "draft_id": draft_dir.name,
                "id": skill.id,
                "version": skill.version,
            })
        return drafts

    def _read_package_files(self, archive_bytes: bytes) -> Dict[str, bytes]:
        if not archive_bytes or len(archive_bytes) > self._MAX_ARCHIVE_BYTES:
            raise SkillValidationError("Skill ZIP 为空或超过大小限制")
        try:
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                files = [info for info in archive.infolist() if not info.is_dir()]
                names = {info.filename for info in files}
                if names != self._PACKAGE_FILES or len(files) != len(self._PACKAGE_FILES):
                    raise SkillValidationError("Skill ZIP 只能包含 skill.json 和 prompt.md")
                if any("/" in name or "\\" in name or name.startswith(".") for name in names):
                    raise SkillValidationError("Skill ZIP 不允许目录或隐藏路径")
                if sum(info.file_size for info in files) > self._MAX_ARCHIVE_BYTES:
                    raise SkillValidationError("Skill ZIP 解压后超过大小限制")
                return {name: archive.read(name) for name in self._PACKAGE_FILES}
        except zipfile.BadZipFile as exc:
            raise SkillValidationError("Skill ZIP 格式非法") from exc
