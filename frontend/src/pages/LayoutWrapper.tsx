import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocation } from '@docusaurus/router';

const LayoutWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { i18n } = useTranslation();
  const location = useLocation();

  useEffect(() => {
    // Set the direction based on the current language
    const direction = i18n.language === 'ur' ? 'rtl' : 'ltr';
    document.documentElement.dir = direction;

    // Set the language attribute
    document.documentElement.lang = i18n.language;
  }, [i18n.language, location.pathname]);

  return <>{children}</>;
};

export default LayoutWrapper;