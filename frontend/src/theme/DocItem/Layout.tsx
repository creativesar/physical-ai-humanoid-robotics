import React from 'react';
import clsx from 'clsx';
import {
  useCurrentSidebarCategory,
} from '@docusaurus/theme-common';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import DocItemMetadata from '@theme/DocItem/Metadata';
import DocItemFooter from '@theme/DocItem/Footer';
import DocItemContent from '@theme/DocItem/Content';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import ChapterPersonalizationBar from '../../components/ChapterPersonalizationBar';
import { usePersonalization } from '../../contexts/PersonalizationContext';

// Fallback imports if the theme components are not available
const FallbackComponent = ({ children, className }: any) => <div className={className}>{children}</div>;
const FallbackBreadcrumbs = () => <div className="breadcrumb-fallback" />;
const FallbackMetadata = () => <></>;
const FallbackFooter = () => <div className="footer-fallback" />;
const FallbackContent = ({ children }: any) => <div>{children}</div>;

export default function DocItemLayout(props: any) {
  const { content: DocContent } = props;
  const { metadata, frontMatter: fm } = useDoc();
  // Cast frontMatter to any to allow custom properties like dir and full_width
  const frontMatter = fm as any;

  // Get personalization context for language
  const { language } = usePersonalization();

  const {
    description,
    title,
    hide_title: hideTitle,
    hide_table_of_contents: hideTableOfContents,
    toc_min_heading_level: tocMinHeadingLevel,
    toc_max_heading_level: tocMaxHeadingLevel,
  } = frontMatter;

  // Only try to access sidebar if it exists in context
  let currentSidebarCategory: any;
  try {
    const sidebarResult = useCurrentSidebarCategory() as any;
    currentSidebarCategory = sidebarResult?.sidebar;
  } catch (e) {
    // If sidebar context is not available, set to undefined
    currentSidebarCategory = undefined;
  }

  // Set direction based on language - Urdu is RTL
  const contentDirection = language === 'urdu' ? 'rtl' : 'ltr';
  const defaultDir = (metadata as any)?.frontMatter?.dir ?? frontMatter.dir ?? contentDirection;

  // Use fallback components if the imported ones are undefined
  const SafeDocItemMetadata = DocItemMetadata || FallbackMetadata;
  const SafeDocBreadcrumbs = DocBreadcrumbs || FallbackBreadcrumbs;
  const SafeDocItemContent = DocItemContent || FallbackContent;
  const SafeDocItemFooter = DocItemFooter || FallbackFooter;

  return (
    <>
      <SafeDocItemMetadata />
      <main
        className={clsx('container margin-vert--lg', {
          'container--full': frontMatter.full_width,
        })}
        dir={contentDirection}
        lang={language === 'urdu' ? 'ur' : 'en'}>
        <div className="row">
          <div className={clsx('col', { 'col--8': !hideTableOfContents })}>
            <div style={{
              position: 'sticky',
              top: '0px',
              zIndex: 1000,
              marginBottom: '1rem'
            }}>
              <ChapterPersonalizationBar />
            </div>
            <SafeDocBreadcrumbs />
            <div className={clsx('margin-vert--lg', {
              'urdu-content': language === 'urdu'
            })} dir={contentDirection}>
              {!hideTitle && <header><h1>{title}</h1></header>}
              <SafeDocItemContent>
                {/* Docusaurus v3 often passes content as props.children or props.content */}
                {DocContent ? <DocContent /> : props.children}
              </SafeDocItemContent>
            </div>
            <SafeDocItemFooter />
          </div>
          {!hideTableOfContents && (
            <div className="col col--2">
              <div
                className="theme-doc-toc-desktop"
                style={{ position: 'sticky', top: '100px' }}
                dir={defaultDir}>
                <div className="table-of-contents__left-border">
                  {DocContent && DocContent.toc && DocContent.toc.length > 0 && (
                    <div
                      className="table-of-contents"
                      role="navigation"
                      aria-label="Table of Contents">
                      <div className="table-of-contents--content">
                        <h3>On this page</h3>
                        {DocContent.toc}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}