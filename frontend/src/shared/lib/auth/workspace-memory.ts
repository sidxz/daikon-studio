// Remembered workspace, so a returning user skips the picker. Survives logout
// on purpose -- the SDK's own workspace id does not -- and "Switch workspace"
// in the header menu clears it to bring the picker back.
//
// The key is namespaced per app, following the suite convention (cellar uses
// `cellar.lastWorkspaceId`, daikon-gen3 `daikon.lastWorkspaceId`).
const KEY = "studio.lastWorkspaceId";

export function rememberedWorkspace(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function rememberWorkspace(id: string): void {
  try {
    localStorage.setItem(KEY, id);
  } catch {
    // Storage unavailable (private mode) -- the picker just shows every time.
  }
}

export function forgetWorkspace(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // Nothing to do; the next sign-in shows the picker.
  }
}
