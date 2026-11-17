/** Query keys for studies and providers, declared once so reads and invalidations cannot drift. */
export const STUDIES_KEY = ["studies"];
export const PROVIDERS_KEY = ["studies", "providers"];

/** Where a study's one conversation is cached. */
export const threadKey = (studyId: string) => [...STUDIES_KEY, studyId, "chat"];
