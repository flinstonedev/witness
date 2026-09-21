from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repository_hygiene.py"
SPEC = importlib.util.spec_from_file_location("repository_hygiene", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hygiene = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hygiene
SPEC.loader.exec_module(hygiene)


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.env = dict(os.environ,
            GIT_AUTHOR_NAME="sample-maintainer",
            GIT_AUTHOR_EMAIL="123+sample-maintainer@users.noreply.github.com",
            GIT_COMMITTER_NAME="sample-maintainer",
            GIT_COMMITTER_EMAIL="123+sample-maintainer@users.noreply.github.com",
            GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        )
        self.git("init", "--initial-branch=main")

    def git(self, *args: str, env: dict[str, str] | None = None, data: bytes | None = None) -> bytes:
        result = subprocess.run(["git", "-C", str(self.repo), *args], input=data,
            env=env or self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(result.returncode, 0, "fixture Git operation failed (details withheld)")
        return result.stdout

    def write(self, name: str, content: str | bytes = "fixture\n") -> None:
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    def commit(self, message: str = "Synthetic fixture", **identity: str) -> str:
        self.git("add", "--all")
        self.git("commit", "--allow-empty", "-m", message, env=dict(self.env, **identity))
        return self.git("rev-parse", "HEAD").decode().strip()

    def findings(self):
        return hygiene.inspect_repository(self.repo)[0]

    def categories(self) -> set[str]:
        return {finding.category for finding in self.findings()}

    def test_public_alias_templates_and_attribution_pass(self) -> None:
        for template in hygiene.ENV_EXAMPLES:
            self.write("examples/" + template.decode())
        self.write("LICENSE", "Copyright Example Scholar <scholar@example.org>\n")
        self.write("AUTHORS", "Research attribution is preserved.\n")
        self.commit("Fixture\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
        self.assertEqual(self.findings(), [])

    def test_github_service_and_bot_alias_pass(self) -> None:
        self.commit(GIT_AUTHOR_NAME="GitHub", GIT_AUTHOR_EMAIL="noreply@github.com",
                    GIT_COMMITTER_NAME="GitHub", GIT_COMMITTER_EMAIL="noreply@github.com")
        self.commit(GIT_AUTHOR_NAME="dependabot[bot]", GIT_AUTHOR_EMAIL="42+dependabot[bot]@users.noreply.github.com")
        self.assertEqual(self.findings(), [])

    def test_private_author_and_committer_email_are_detected(self) -> None:
        self.commit(GIT_AUTHOR_EMAIL="writer@example.org", GIT_COMMITTER_EMAIL="builder@example.org")
        findings = self.findings()
        self.assertEqual({f.location for f in findings}, {"author", "committer"})
        self.assertEqual(self.categories(), {"nonpublic_identity_email"})

    def test_full_name_with_noreply_address_is_detected(self) -> None:
        self.commit(GIT_AUTHOR_NAME="Example Private Person")
        self.assertEqual(self.categories(), {"nonpublic_identity_name"})

    def test_private_name_only_in_committer_is_detected(self) -> None:
        self.commit(GIT_COMMITTER_NAME="Example Private Person")
        self.assertEqual([(f.category, f.location) for f in self.findings()], [("nonpublic_identity_name", "committer")])

    def test_malformed_noreply_domain_cannot_smuggle_an_identity(self) -> None:
        header = b"author sample-maintainer <sample-maintainer@private-value@users.noreply.github.com> 1000000000 +0000"
        self.assertEqual([f.category for f in hygiene.identity_findings("0" * 40, header)], ["nonpublic_identity_email"])

    def test_deleted_private_config_and_same_blob_safe_path_are_detected(self) -> None:
        self.write(".env", "same bytes\n")
        self.write(".env.example", "same bytes\n")
        self.commit()
        (self.repo / ".env").unlink()
        self.commit("Remove local file")
        self.assertIn("private_configuration_path", self.categories())

    def test_nested_backups_overrides_and_odd_filenames_are_detected(self) -> None:
        for name in ["nested/.private/backup.json", "ops/wrangler.local.jsonc", "ops/service.private.jsonc", "nested/.dev.vars", "nested/.env.production", "odd\tname\npath/.env"]:
            self.write(name)
        self.commit()
        self.assertEqual(len(self.findings()), 6)
        self.assertEqual(self.categories(), {"private_configuration_path"})

    def test_symlink_named_as_private_config_is_detected(self) -> None:
        (self.repo / ".env").symlink_to("safe-target")
        self.commit()
        self.assertIn("private_configuration_path", self.categories())

    def test_posix_windows_and_binary_embedded_home_paths_are_detected(self) -> None:
        # Join segments so this test's source contains no actual home literal.
        posix = "/" + "Users" + "/fixture-person/private.txt"
        linux = "/" + "home" + "/fixture-person"
        windows = "C:" + "\\" + "Users" + "\\fixture-person\\private.txt"
        self.write("paths.txt", "\n".join([posix, linux, windows, json.dumps(windows)]))
        self.write("binary.bin", b"\0" + posix.encode())
        self.commit()
        self.assertEqual(len(self.findings()), 5)
        self.assertEqual(self.categories(), {"absolute_user_home_path"})

    def test_private_session_forms_and_trailers_are_detected(self) -> None:
        locations = [
            ("chatgpt.com", "/c/fixture-id"),
            ("chat.openai.com", "/c/fixture-id"),
            ("chatgpt.com", "/g/g-example/c/fixture-id"),
            ("chatgpt.com", "/codex/tasks/fixture-id"),
            ("claude.ai", "/chat/fixture-id"),
            ("claude.ai", "/code/session_fixture-id"),
        ]
        urls = ["https://" + host + path for host, path in locations]
        self.commit("Fixture\n" + "\n".join(urls) + "\n\nclaude-code-session: fixture-id\nCodex-Thread-Id: fixture-id")
        findings = self.findings()
        self.assertEqual(sum(f.category == "private_assistant_session_url" for f in findings), 6)
        self.assertEqual(sum(f.category == "private_assistant_session_trailer" for f in findings), 2)

    def test_public_docs_share_and_anthropic_credit_are_not_sessions(self) -> None:
        urls = ["https://" + "chatgpt.com" + "/share/public-fixture", "https://" + "claude.ai" + "/share/public-fixture", "https://docs.anthropic.com/"]
        self.commit("Public references\n" + "\n".join(urls) + "\nCo-Authored-By: Claude <noreply@anthropic.com>")
        self.assertEqual(self.findings(), [])

    def test_deleted_session_blob_is_still_checked(self) -> None:
        url = "https://" + "claude.ai" + "/chat/fixture-id"
        self.write("notes.md", url + "\nCodex-Session: fixture-id\n")
        self.commit()
        (self.repo / "notes.md").unlink()
        self.commit("Remove saved session")
        self.assertEqual(self.categories(), {"private_assistant_session_url", "private_assistant_session_trailer"})

    def test_non_main_branch_and_annotated_tag_are_checked(self) -> None:
        first = self.commit()
        self.git("checkout", "-b", "other")
        self.write(".dev.vars", "fixture\n")
        self.commit(GIT_AUTHOR_EMAIL="writer@example.org")
        self.git("checkout", "main")
        self.git("tag", "-a", "fixture-tag", first, "-m", "Codex-Session: fixture-id")
        self.assertTrue({"private_configuration_path", "nonpublic_identity_email", "private_assistant_session_trailer"} <= self.categories())

    def test_git_replace_cannot_hide_original_identity(self) -> None:
        original = self.commit(GIT_AUTHOR_EMAIL="writer@example.org")
        tree = self.git("rev-parse", "HEAD^{tree}").decode().strip()
        replacement = self.git("commit-tree", tree, "-m", "Public replacement").decode().strip()
        self.git("replace", original, replacement)
        self.assertIn("nonpublic_identity_email", self.categories())

    def test_output_never_contains_matching_values_or_filenames(self) -> None:
        marker = "fixture-sensitive-person"
        home = "/" + "Users" + "/" + marker + "/file"
        self.write(marker + "/.env", home)
        self.commit(GIT_AUTHOR_EMAIL=marker + "@example.org")
        result = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(self.repo), "--json"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn(marker, result.stdout + result.stderr)
        self.assertNotIn(str(self.repo), result.stdout + result.stderr)
        self.assertNotIn(".env", result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["findings"])

    def test_checker_and_tests_do_not_need_blanket_fixture_exclusions(self) -> None:
        self.write("scripts/check_repository_hygiene.py", SCRIPT.read_bytes())
        self.write("tests/test_repository_hygiene.py", Path(__file__).read_bytes())
        self.commit()
        self.assertEqual(self.findings(), [])

    def test_shallow_history_is_rejected(self) -> None:
        self.commit()
        self.commit()
        shallow = Path(self.temporary.name) / "shallow"
        subprocess.run(["git", "clone", "--depth=1", self.repo.as_uri(), str(shallow)], env=self.env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with self.assertRaises(hygiene.InspectionError):
            hygiene.inspect_repository(shallow)


if __name__ == "__main__":
    unittest.main()
