/**
 * Root query keys for the engines feature, declared once. Re-declaring a key
 * per file is how reads and invalidations silently drift apart.
 */
export const ENGINES_KEY = ["engines"];
