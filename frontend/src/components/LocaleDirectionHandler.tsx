import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const LocaleDirectionHandler: React.FC = () => {
  const { i18n } = useTranslation();

  useEffect(() => {
    // Set the direction based on the current language
    const direction = i18n.language === 'ur' ? 'rtl' : 'ltr';
    document.documentElement.dir = direction;

    // Set the language attribute
    document.documentElement.lang = i18n.language;
  }, [i18n.language]);

  return null;
};

export default LocaleDirectionHandler;