import React from 'react';
import { I18nextProvider } from 'react-i18next';
import i18n from './i18n/i18n';
import LocaleDirectionHandler from './components/LocaleDirectionHandler';

// Root component to wrap the entire application with i18n provider
export default function Root({ children }) {
  return (
    <I18nextProvider i18n={i18n}>
      <LocaleDirectionHandler />
      {children}
    </I18nextProvider>
  );
}