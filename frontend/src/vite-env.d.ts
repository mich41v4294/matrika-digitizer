/// <reference types="vite/client" />

declare const __BUILD_COMMIT_HASH__: string;

declare module "node:child_process" {
  export function execSync(
    command: string,
    options?: { encoding?: string; stdio?: string | string[]; cwd?: string }
  ): string;
}

declare module "node:fs" {
  export function existsSync(path: string): boolean;
  export function readFileSync(path: string, encoding: string): string;
  export function statSync(path: string): { isDirectory(): boolean; isFile(): boolean };
}

declare module "node:path" {
  export function resolve(...paths: string[]): string;
  export function dirname(path: string): string;
  export function join(...paths: string[]): string;
}
