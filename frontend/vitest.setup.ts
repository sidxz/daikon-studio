import "@testing-library/jest-dom";

// Radix UI uses ResizeObserver internally; jsdom does not ship one.
if (typeof global.ResizeObserver === "undefined") {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

// cmdk scrolls its active item into view.
if (typeof HTMLElement !== "undefined" && !HTMLElement.prototype.scrollIntoView) {
  HTMLElement.prototype.scrollIntoView = () => {};
}

if (typeof window !== "undefined") {
  if (!window.matchMedia) {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }),
    });
  }

  // Node 25 ships a built-in Web Storage `localStorage` that shadows jsdom's
  // before jsdom can serve its own. Without `--localstorage-file` its getter
  // warns and hands back an object with no `.clear()`, so any test that resets
  // storage between cases blows up. Override with a real in-memory store.
  const createInMemoryStorage = () => {
    let store: Record<string, string> = {};
    return {
      getItem: (key: string): string | null => store[key] ?? null,
      setItem: (key: string, value: string): void => {
        store[key] = String(value);
      },
      removeItem: (key: string): void => {
        delete store[key];
      },
      clear: (): void => {
        store = {};
      },
      get length(): number {
        return Object.keys(store).length;
      },
      key: (index: number): string | null => Object.keys(store)[index] ?? null,
    };
  };
  Object.defineProperty(window, "localStorage", {
    value: createInMemoryStorage(),
    writable: true,
  });
  Object.defineProperty(window, "sessionStorage", {
    value: createInMemoryStorage(),
    writable: true,
  });
}
