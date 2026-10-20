/**
 * Formatting for the build a runner's agent reported at start.
 *
 * Nothing here infers anything. The agent sends its version, commit and build
 * date once when it starts (`POST /api/v1/runner/identity`); a runner whose
 * agent has not started since the studio began asking sends nothing at all, and
 * every field arrives null. "unknown" is also a legitimate reported value --
 * that is what the backend's `build_info()` returns for a field the image was
 * built without -- so it is shown as given rather than hidden.
 */

/** The hover line: the detail an operator wants only once they are suspicious. */
export function buildDetail(runner: {
  build_git_sha: string | null;
  build_date: string | null;
}): string {
  return `Commit ${runner.build_git_sha ?? "unknown"}, built ${runner.build_date ?? "unknown"}`;
}

/** Git's own abbreviation. A full 40-character sha in a table cell is noise;
 *  anything that is not one (a short sha, "unknown") is left alone. */
export function shortSha(sha: string): string {
  return /^[0-9a-f]{40}$/i.test(sha) ? sha.slice(0, 7) : sha;
}
