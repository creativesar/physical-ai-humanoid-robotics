import React from 'react';
import { I18nextProvider } from 'react-i18next';
import i18n from '../i18n/i18n';
import LocaleDirectionHandler from './LocaleDirectionHandler';

// Root component to wrap the entire application with i18n provider
const Root: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <I18nextProvider i18n={i18n}>
      <LocaleDirectionHandler />
      {children}
    </I18nextProvider>
  );
};

export default Root;