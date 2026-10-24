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
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function readFileSync(path: string, encoding: string): string;
  export function statSync(path: string): { isDirectory(): boolean; isFile(): boolean };
  export function writeFileSync(path: string, data: string, encoding?: string): void;
}

declare module "node:https" {
  export interface IncomingMessage {
    statusCode?: number;
    setEncoding(encoding: string): void;
    on(event: "data", listener: (chunk: string) => void): this;
    on(event: "end", listener: () => void): this;
    on(event: "error", listener: (err: unknown) => void): this;
  }
  export interface ClientRequest {
    on(event: "error", listener: (err: unknown) => void): this;
    on(event: "timeout", listener: () => void): this;
    destroy(): void;
  }
  export function get(
    url: string,
    options: { headers?: Record<string, string>; timeout?: number },
    callback: (res: IncomingMessage) => void
  ): ClientRequest;
}

declare module "node:path" {
  export function resolve(...paths: string[]): string;
  export function dirname(path: string): string;
  export function join(...paths: string[]): string;
}
