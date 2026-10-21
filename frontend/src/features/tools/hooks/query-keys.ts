/**
 * Root query keys for the tools feature, declared once. Re-declaring a key per
 * file is how reads and invalidations silently drift apart.
 */
export const TOOLS_KEY = ["tools"];
