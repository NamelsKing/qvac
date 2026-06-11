'use client';

import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';
import { Sparkles, User } from 'lucide-react';
import { isValidElement, type ReactElement, type ReactNode } from 'react';
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
            // raw text off it, then hand it to Fumadocs'
            // `DynamicCodeBlock`, which Shiki-highlights at runtime
            // (matching the docs' own code blocks) and ships a copy
            // button. `not-prose` keeps the global `.prose code`
            // inline-pill styling from leaking onto the highlighted
            // tokens.
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
            return (
              <div className="not-prose my-2 text-xs">
                <DynamicCodeBlock lang={language ?? 'text'} code={code} />
              </div>
            );
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
