import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .manifest import Manifest, calculate_hash


DEFAULT_TIMEOUT = 30
DEFAULT_RATE_LIMIT_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class RateLimitPolicy:
    delay_seconds: float = DEFAULT_RATE_LIMIT_DELAY
    max_retries: int = DEFAULT_MAX_RETRIES


def _sleep(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _request_json(
    session: requests.Session,
    url: str,
    logger,
    params: Optional[Dict[str, Any]] = None,
    policy: Optional[RateLimitPolicy] = None,
    timeout: int = DEFAULT_TIMEOUT,
    allow_404: bool = False,
) -> Optional[Dict[str, Any]]:
    policy = policy or RateLimitPolicy()
    for attempt in range(1, policy.max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in {429} or response.status_code >= 500:
                logger.error(
                    "Transient error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            if response.status_code == 404:
                if not allow_404:
                    body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                    logger.error(
                        "Client error from source.",
                        url=url,
                        status=response.status_code,
                        attempt=attempt,
                        content_type=response.headers.get("Content-Type"),
                        server=response.headers.get("Server"),
                        cache=response.headers.get("X-Cache"),
                        drupal_cache=response.headers.get("X-Drupal-Cache"),
                        request_id=response.headers.get("X-Request-Id"),
                        body_preview=body_preview,
                    )
                break
            if 400 <= response.status_code < 500:
                body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                logger.error(
                    "Client error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                    content_type=response.headers.get("Content-Type"),
                    server=response.headers.get("Server"),
                    cache=response.headers.get("X-Cache"),
                    drupal_cache=response.headers.get("X-Drupal-Cache"),
                    request_id=response.headers.get("X-Request-Id"),
                    body_preview=body_preview,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            response.raise_for_status()
            _sleep(policy.delay_seconds)
            return response.json()
        except requests.RequestException as exc:
            logger.error("Request failed.", url=url, error=str(exc), attempt=attempt)
            _sleep(policy.delay_seconds * attempt)
    return None


def _request_text(
    session: requests.Session,
    url: str,
    logger,
    policy: Optional[RateLimitPolicy] = None,
    timeout: int = DEFAULT_TIMEOUT,
    allow_404: bool = False,
) -> Optional[str]:
    policy = policy or RateLimitPolicy()
    for attempt in range(1, policy.max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429} or response.status_code >= 500:
                logger.error(
                    "Transient error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            if response.status_code == 404:
                if not allow_404:
                    body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                    logger.error(
                        "Client error from source.",
                        url=url,
                        status=response.status_code,
                        attempt=attempt,
                        content_type=response.headers.get("Content-Type"),
                        server=response.headers.get("Server"),
                        cache=response.headers.get("X-Cache"),
                        drupal_cache=response.headers.get("X-Drupal-Cache"),
                        request_id=response.headers.get("X-Request-Id"),
                        body_preview=body_preview,
                    )
                break
            if 400 <= response.status_code < 500:
                body_preview = (response.text or "").strip().replace("\n", " ")[:300]
                logger.error(
                    "Client error from source.",
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                    content_type=response.headers.get("Content-Type"),
                    server=response.headers.get("Server"),
                    cache=response.headers.get("X-Cache"),
                    drupal_cache=response.headers.get("X-Drupal-Cache"),
                    request_id=response.headers.get("X-Request-Id"),
                    body_preview=body_preview,
                )
                _sleep(policy.delay_seconds * attempt)
                continue
            response.raise_for_status()
            _sleep(policy.delay_seconds)
            return response.text
        except requests.RequestException as exc:
            logger.error("Request failed.", url=url, error=str(exc), attempt=attempt)
            _sleep(policy.delay_seconds * attempt)
    return None


def _is_drupal_core_11(constraint: str) -> bool:
    if not constraint:
        return False
    return re.search(r"(^|[^\d])\^?11(\.|$)", constraint) is not None


def _extract_machine_name(node: Dict[str, Any]) -> Optional[str]:
    for key in ("field_project_machine_name", "machine_name", "project_machine_name"):
        value = node.get(key)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        if isinstance(value, str):
            return value
    return None


def _extract_default_branch(node: Dict[str, Any]) -> Optional[str]:
    value = node.get("field_project_default_branch")
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if isinstance(value, str):
        return value
    return None


def _extract_changed_epoch(node: Dict[str, Any]) -> Optional[int]:
    for key in ("changed", "field_last_modified", "created"):
        value = node.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            continue
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            continue
    return None


def _extract_project_usage(node: Dict[str, Any]) -> int:
    usage = node.get("project_usage", {})
    if not isinstance(usage, dict):
        return 0
    total = 0
    for value in usage.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _iter_drupal_project_nodes(
    session: requests.Session,
    endpoint: str,
    logger,
    policy: RateLimitPolicy,
    max_pages: int = 1,
    limit: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    sort: str = "changed",
    direction: str = "desc",
) -> Iterable[Dict[str, Any]]:
    for page in range(max_pages):
        params = {
            "type": "project_module",
            "status": 1,
            "sort": sort,
            "direction": direction,
        }
        if filters:
            for key, value in filters.items():
                if key in {"field_project_machine_name", "nid"}:
                    params[key] = value
        if page > 0:
            params["page"] = page
        if limit:
            params["limit"] = limit

        payload = _request_json(
            session,
            endpoint,
            logger,
            params=params,
            policy=policy,
        )
        if not payload or "list" not in payload:
            break
        entries = payload.get("list", [])
        if not entries:
            break
        for node in entries:
            yield node


def _discover_drupal_projects(
    session: requests.Session,
    endpoint: str,
    logger,
    policy: RateLimitPolicy,
    max_pages: int = 1,
    limit: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    sort: str = "changed",
    direction: str = "desc",
) -> List[Dict[str, Any]]:
    return list(
        _iter_drupal_project_nodes(
            session,
            endpoint,
            logger,
            policy,
            max_pages=max_pages,
            limit=limit,
            filters=filters,
            sort=sort,
            direction=direction,
        )
    )


def _parse_version_pairs(constraint: str) -> list[tuple[int, int]]:
    matches = re.findall(r"(\d+)\.(\d+)", constraint or "")
    return [(int(major), int(minor)) for major, minor in matches]


def _min_version_pair(constraint: str) -> Optional[tuple[int, int]]:
    pairs = _parse_version_pairs(constraint)
    if not pairs:
        return None
    return min(pairs)


def _supports_php_min(constraint: str, minimum: str) -> bool:
    if not constraint:
        return False
    required = _min_version_pair(minimum)
    found = _min_version_pair(constraint)
    if not required or not found:
        return False
    return found >= required


def _looks_archived_or_security_only(project: Dict[str, Any]) -> bool:
    title = str(project.get("title", "")).lower()
    machine_name = str(_extract_machine_name(project) or "").lower()
    text = " ".join([title, machine_name])
    archived_markers = ["archived", "deprecated", "security only", "end of life", "eol"]
    return any(marker in text for marker in archived_markers)


def _fetch_composer_metadata(
    session: requests.Session,
    logger,
    machine_name: str,
    branches: List[str],
    policy: RateLimitPolicy,
) -> Optional[Dict[str, Any]]:
    for branch in branches:
        url = f"https://git.drupalcode.org/project/{machine_name}/raw/{branch}/composer.json"
        text = _request_text(session, url, logger, policy=policy, allow_404=True)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Invalid composer.json content.", url=url)
            return None

        require = data.get("require", {}) if isinstance(data.get("require"), dict) else {}
        require_dev = data.get("require-dev", {}) if isinstance(data.get("require-dev"), dict) else {}
        core_constraint = require.get("drupal/core") or require.get("drupal/core-recommended")
        php_constraint = require.get("php")
        has_tests = any("phpunit" in key.lower() for key in require_dev.keys())

        return {
            "constraint": core_constraint,
            "php_constraint": php_constraint,
            "branch": branch,
            "url": url,
            "has_tests": has_tests,
        }
    return None


def _build_curated_sources(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    drupal_core = config.get("sources", {}).get("drupal_core", {})
    return [
        {
            "id": "drupal_core",
            "type": drupal_core.get("type", "git"),
            "url": drupal_core.get("url"),
            "ref": drupal_core.get("branch"),
            "description": "Drupal core repository",
        },
        {
            "id": "drupal_docs",
            "type": "http",
            "url": "https://www.drupal.org/docs",
            "ref": None,
            "description": "Drupal official documentation",
        },
        {
            "id": "drupal_api",
            "type": "http",
            "url": "https://api.drupal.org/api/drupal",
            "ref": None,
            "description": "Drupal API reference",
        },
        {
            "id": "symfony_docs",
            "type": "http",
            "url": "https://symfony.com/doc/7.0/index.html",
            "ref": None,
            "description": "Symfony 7 documentation",
        },
        {
            "id": "drupal_security",
            "type": "http",
            "url": "https://www.drupal.org/security",
            "ref": None,
            "description": "Drupal security advisories",
        },
    ]


def _score_project(
    *,
    is_drupal11: bool,
    has_php_83: bool,
    recent_bonus: int,
    has_tests: bool,
    usage: int,
    machine_name: str,
) -> tuple[int, int, int, int, int, str]:
    # Ranking key: Drupal 11 > PHP 8.3 > recent changed > has tests, then usage and id tie-breakers.
    return (
        1 if is_drupal11 else 0,
        1 if has_php_83 else 0,
        recent_bonus,
        1 if has_tests else 0,
        usage,
        machine_name,
    )


def _write_sources_manifest(
    path: Path,
    seed: Optional[int],
    curated_sources: List[Dict[str, Any]],
    drupal_sources: List[Dict[str, Any]],
    policy: RateLimitPolicy,
    metrics: Dict[str, Any],
) -> None:
    manifest = {
        "stage": "sources",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "rate_limit": {
            "delay_seconds": policy.delay_seconds,
            "max_retries": policy.max_retries,
        },
        "metrics": metrics,
        "sources": {
            "curated": curated_sources,
            "drupal_org_projects": drupal_sources,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_sources_stage(config: Dict[str, Any], logger, root: Path) -> int:
    sources_dir = root / config.get("directories", {}).get("sources", "sources")
    seed = int(config.get("seed", 42))
    rng = random.Random(seed)

    policy = RateLimitPolicy()
    session = requests.Session()

    source_cfg = config.get("sources", {})
    drupal_project_config = source_cfg.get("drupal_projects", {})
    discovery_cfg = source_cfg.get("discovery", {})
    filter_cfg = source_cfg.get("filters", {})

    user_agent = drupal_project_config.get("user_agent") or "DrupalGym/1.0"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    from_header = drupal_project_config.get("from")
    referer = drupal_project_config.get("referer")
    extra_headers = drupal_project_config.get("headers", {})
    if from_header:
        headers["From"] = from_header
    if referer:
        headers["Referer"] = referer
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    session.headers.update(headers)

    curated_sources = _build_curated_sources(config)

    drupal_endpoint = drupal_project_config.get("endpoint")
    discovered_projects: List[Dict[str, Any]] = []
    filtered_projects: List[Dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}

    if drupal_endpoint:
        logger.info("Discovering Drupal.org projects.", endpoint=drupal_endpoint)
        max_pages = int(discovery_cfg.get("max_pages", drupal_project_config.get("max_pages", 5)))
        limit = int(discovery_cfg.get("limit_per_page", drupal_project_config.get("limit", 100)))
        filters = drupal_project_config.get("filters", {})
        sort = drupal_project_config.get("sort", "changed")
        direction = drupal_project_config.get("direction", "desc")

        discovered_projects = _discover_drupal_projects(
            session,
            drupal_endpoint,
            logger,
            policy,
            max_pages=max_pages,
            limit=limit,
            filters=filters,
            sort=sort,
            direction=direction,
        )
        logger.info("Discovered Drupal.org projects.", count=len(discovered_projects))

    min_recent_days = int(discovery_cfg.get("min_last_changed_days", 365))
    max_after_filter = int(discovery_cfg.get("max_projects_after_filter", 200))
    require_core_constraint = str(filter_cfg.get("require_drupal_core_constraint", "^11"))
    require_php_min = str(filter_cfg.get("require_php_constraint_min", "8.3"))
    exclude_archived = bool(filter_cfg.get("exclude_archived_or_security_only", True))

    candidates: list[dict[str, Any]] = []

    for project in discovered_projects:
        machine_name = _extract_machine_name(project)
        if not machine_name:
            rejection_reasons["missing_machine_name"] = rejection_reasons.get("missing_machine_name", 0) + 1
            continue

        if exclude_archived and _looks_archived_or_security_only(project):
            rejection_reasons["archived_or_security_only"] = rejection_reasons.get("archived_or_security_only", 0) + 1
            continue

        default_branch = _extract_default_branch(project)
        candidate_branches = [default_branch] if default_branch else []
        for branch in ["11.x", "11.0.x", "1.x", "1.0.x", "2.x", "3.x", "main", "master"]:
            if branch and branch not in candidate_branches:
                candidate_branches.append(branch)

        composer_info = _fetch_composer_metadata(session, logger, machine_name, candidate_branches, policy)
        if not composer_info:
            rejection_reasons["missing_composer"] = rejection_reasons.get("missing_composer", 0) + 1
            continue

        core_constraint = str(composer_info.get("constraint") or "")
        php_constraint = str(composer_info.get("php_constraint") or "")

        is_drupal11 = _is_drupal_core_11(core_constraint)
        if require_core_constraint and not is_drupal11:
            rejection_reasons["core_not_drupal11"] = rejection_reasons.get("core_not_drupal11", 0) + 1
            continue

        has_php_83 = _supports_php_min(php_constraint, require_php_min)
        if require_php_min and not has_php_83:
            rejection_reasons["php_constraint_too_low"] = rejection_reasons.get("php_constraint_too_low", 0) + 1
            continue

        changed_epoch = _extract_changed_epoch(project)
        last_changed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(changed_epoch)) if changed_epoch else None
        last_changed_days = None
        recent_bonus = 0
        if changed_epoch:
            last_changed_days = int((time.time() - changed_epoch) // 86400)
            if last_changed_days <= max(1, min_recent_days):
                recent_bonus = 1

        usage = _extract_project_usage(project)
        has_tests = bool(composer_info.get("has_tests", False))

        selection_reason = (
            f"drupal_core={core_constraint or 'missing'}; "
            f"php={php_constraint or 'missing'}; "
            f"recent={recent_bonus == 1}; has_tests={has_tests}"
        )

        score_tuple = _score_project(
            is_drupal11=is_drupal11,
            has_php_83=has_php_83,
            recent_bonus=recent_bonus,
            has_tests=has_tests,
            usage=usage,
            machine_name=machine_name,
        )

        candidates.append(
            {
                "id": machine_name,
                "type": "git",
                "url": f"https://git.drupalcode.org/project/{machine_name}.git",
                "ref": composer_info.get("branch"),
                "composer_url": composer_info.get("url"),
                "composer_constraints": {
                    "drupal_core": core_constraint,
                    "php": php_constraint,
                },
                "core_constraint": core_constraint,
                "php_constraint": php_constraint,
                "last_changed": last_changed,
                "last_changed_days": last_changed_days,
                "has_tests": has_tests,
                "usage": usage,
                "selection_reason": selection_reason,
                "_score": score_tuple,
            }
        )

    # Deterministic ranking with seed-based tiebreak.
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item["_score"], reverse=True)

    for entry in candidates[:max_after_filter]:
        clean_entry = dict(entry)
        clean_entry.pop("_score", None)
        filtered_projects.append(clean_entry)

    metrics = {
        "projects_discovered": len(discovered_projects),
        "projects_candidates": len(candidates),
        "projects_filtered": len(filtered_projects),
        "curated_sources": len(curated_sources),
        "rejection_reasons": rejection_reasons,
        "max_projects_after_filter": max_after_filter,
    }

    logger.metric("projects_discovered", len(discovered_projects))
    logger.metric("projects_candidates", len(candidates))
    logger.metric("projects_filtered", len(filtered_projects))
    logger.metric("curated_sources", len(curated_sources))

    manifest_path = sources_dir / "manifest.json"
    _write_sources_manifest(
        manifest_path,
        seed,
        curated_sources,
        filtered_projects,
        policy,
        metrics,
    )

    sources_manifest = Manifest("sources", sources_dir)
    sources_manifest.set_metrics(metrics)
    manifest_hash = calculate_hash(manifest_path)
    sources_manifest.add_output(
        "sources_manifest",
        str(manifest_path.relative_to(root)),
        manifest_hash,
    )
    sources_manifest.save(root / "manifests" / "sources_manifest.json")

    logger.info("Wrote sources manifest.", path=str(manifest_path))
    return 0
