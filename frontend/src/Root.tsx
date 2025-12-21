import React from 'react';

// Root component to wrap the entire application
const Root: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <>{children}</>;
};

export default Root;