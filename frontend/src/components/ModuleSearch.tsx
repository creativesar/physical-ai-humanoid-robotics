import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

interface SearchResult {
  title: string;
  content: string;
  url: string;
  section: string;
}

const ModuleSearch: React.FC = () => {
  const location = useLocation();
  const { siteConfig } = useDocusaurusContext();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Extract current module from URL
  const pathParts = location.pathname.split('/').filter(part => part);
  const currentModule = pathParts[0] || 'intro';

  // In a real implementation, this would connect to the backend search API
  // For now, we'll create a mock search implementation
  const mockSearch = (query: string): SearchResult[] => {
    if (!query) return [];
    
    // This is a simplified mock implementation
    // In a real app, this would query the backend /query endpoint
    return [
      {
        title: "Sample Search Result",
        content: `Results for "${query}" in module ${currentModule}`,
        url: location.pathname,
        section: "Sample Section"
      }
    ];
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    
    // Simulate API call delay
    setTimeout(() => {
      const results = mockSearch(searchQuery);
      setSearchResults(results);
      setIsSearching(false);
      setShowResults(true);
    }, 300);
  };

  // Close results when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      if (showResults && searchQuery === '') {
        setShowResults(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [showResults, searchQuery]);

  return (
    <div className="module-search-container">
      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search within this module..."
          className="search-input"
          onFocus={() => searchQuery && setShowResults(true)}
        />
        <button type="submit" className="search-button" disabled={isSearching}>
          {isSearching ? 'Searching...' : 'Search'}
        </button>
      </form>

      {showResults && (
        <div className="search-results-panel">
          {searchResults.length > 0 ? (
            <ul className="search-results-list">
              {searchResults.map((result, index) => (
                <li key={index} className="search-result-item">
                  <a href={result.url} className="search-result-link">
                    <h4>{result.title}</h4>
                    <p>{result.content}</p>
                    <span className="search-result-section">Section: {result.section}</span>
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <div className="no-results">No results found for "{searchQuery}"</div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModuleSearch;