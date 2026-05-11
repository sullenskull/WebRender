/// <reference types="vite/client" />

declare module '*.wgsl' {
  const shader: string;
  export default shader;
}

declare module '*.hdr' {
  const url: string;
  export default url;
}
