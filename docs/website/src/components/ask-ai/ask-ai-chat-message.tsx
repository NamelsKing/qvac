'use client';

import { Check, Copy, Sparkles, User } from 'lucide-react';
import {
  isValidElement,
  useCallback,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { cn } from '@/lib/cn';
import { AskAIReferences } from './ask-ai-references';
import type { ChatMessage } from './use-ask-ai-chat';

/**
 * A single chat message. User messages render as plain text (we don't
 * want to interpret what the user typed as markdown). Assistant
 * messages render as markdown via `react-markdown` + `remark-gfm` so
 * Inkeep's GitHub-flavored markdown output (links, tables, code
 * fences) lays out correctly.
 *
 * Layout mirrors a typical docs-assistant pattern: a small role
 * indicator on the left, then the message body. The body owns its
 * own max-width so long messages wrap without horizontally
 * stretching the modal.
 */
export function AskAIChatMessage({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  return (
    <div
      data-role={message.role}
      className={cn(
        'flex gap-3 px-4 py-3',
        isUser ? 'bg-fd-muted/30' : 'bg-transparent',
      )}
    >
      <div
        aria-hidden="true"
        className={cn(
          'mt-0.5 inline-flex size-6 shrink-0 items-center justify-center rounded-full border',
          isUser
            ? 'bg-fd-popover text-fd-muted-foreground'
            : 'bg-fd-primary/10 text-fd-primary',
        )}
      >
        {isUser ? <User className="size-3.5" /> : <Sparkles className="size-3.5" />}
      </div>
      <div className="min-w-0 flex-1 text-sm leading-relaxed text-fd-popover-foreground">
        {isUser ? (
          // Preserve the user's whitespace (including newlines from
          // pasted snippets) without invoking the markdown parser.
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <>
            <MarkdownBody content={message.content} />
            {message.references && message.references.length > 0 ? (
              <AskAIReferences references={message.references} />
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Assistant-message markdown renderer. We keep the rule list small
 * and apply targeted Tailwind classes so the output picks up the
 * fumadocs design tokens (`text-fd-primary`, `bg-fd-card`, etc.)
 * without dragging in a full `@tailwindcss/typography` plugin.
 *
 * Streaming-friendly: react-markdown re-parses on every render, but
 * the message body for a single assistant turn is at most a few
 * KB so this is fine for the v1 cost.
 */
function MarkdownBody({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none break-words text-fd-popover-foreground prose-headings:text-fd-popover-foreground prose-strong:text-fd-popover-foreground prose-code:text-fd-primary">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...rest }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer noopener"
              className="text-fd-primary underline underline-offset-2 hover:text-fd-primary/80"
              {...rest}
            >
              {children}
            </a>
          ),
          // The `code` override is now ONLY ever reached for inline
          // code. Block code is intercepted by the `pre` override
          // below, which reads the fenced `<code>` element's props
          // directly and renders its own markup - so react-markdown
          // never routes the block's inner `<code>` through here.
          // That makes the inline-vs-block split deterministic
          // (remark always wraps block code in `<pre>`; inline code
          // never is) and fixes fences that omit a language.
          code: ({ node: _node, className: _className, children, ...rest }) => (
            <code
              className="rounded bg-fd-muted px-1 py-0.5 font-mono text-[0.85em] text-fd-primary"
              {...rest}
            >
              {children}
            </code>
          ),
          pre: ({ children }) => {
            // `children` is the `<code>` element react-markdown built
            // for the fence. Pull the language (`language-xxx`) and the
            // raw text off it, then render a self-contained block.
            const codeEl = isValidElement(children)
              ? (children as ReactElement<{
                  className?: string;
                  children?: ReactNode;
                }>)
              : null;
            if (!codeEl) {
              return (
                <pre className="my-2 overflow-x-auto rounded-md border bg-fd-muted/40 p-3 font-mono text-xs">
                  {children}
                </pre>
              );
            }
            const language = /language-(\w+)/.exec(
              codeEl.props.className ?? '',
            )?.[1];
            const code = nodeToText(codeEl.props.children).replace(/\n$/, '');
            return <ChatCodeBlock language={language} code={code} />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Flatten an arbitrary React node tree into its plain text. Used to
 * recover the raw source of a fenced code block from the `<code>`
 * element react-markdown hands to our `pre` override, so we can feed
 * it to the copy button verbatim.
 */
function nodeToText(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(nodeToText).join('');
  if (isValidElement(node)) {
    return nodeToText((node.props as { children?: ReactNode }).children);
  }
  return '';
}

/**
 * A fenced code block in an assistant message. Distinct from inline
 * code: it gets its own card with a language label and a copy button,
 * and a horizontally scrollable body so long lines don't stretch the
 * modal. Renders a native `<pre><code>` so it is NOT re-routed through
 * the `code` component override (which only styles inline code).
 */
function ChatCodeBlock({ language, code }: { language?: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    if (typeof navigator === 'undefined' || !navigator.clipboard) return;
    void navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  }, [code]);

  return (
    // `not-prose` opts the whole block out of the `.prose` wrapper's
    // typography rules (see `MarkdownBody`). Without it, the global
    // `.prose code` rule in global.css paints the INLINE-code box
    // (background + border) onto the block's own `<code>`, so the
    // fenced code reads as inline code nested in a block. Opting out
    // keeps the mono font on the block itself, undecorated.
    <div className="not-prose my-2 overflow-hidden rounded-md border bg-fd-muted/40">
      <div className="flex items-center justify-between gap-2 border-b border-fd-border/60 bg-fd-muted/60 px-3 py-1">
        <span className="font-mono text-[0.7rem] lowercase tracking-wide text-fd-muted-foreground">
          {language ?? 'code'}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          title={copied ? 'Copied' : 'Copy code'}
          className="inline-flex size-6 shrink-0 items-center justify-center rounded text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fd-ring"
        >
          {copied ? (
            <Check className="size-3.5" aria-hidden="true" />
          ) : (
            <Copy className="size-3.5" aria-hidden="true" />
          )}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}
