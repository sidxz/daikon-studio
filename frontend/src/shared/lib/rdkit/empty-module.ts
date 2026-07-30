// RDKit's WASM loader has a `require('fs')` on its Node detection path. The
// browser build reaches the binary over fetch, so `fs` is aliased here in
// next.config.ts and this is what it resolves to.
export default {};
