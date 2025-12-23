import React, { useState, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useLocation } from '@docusaurus/router';

interface ProgressData {
  [moduleId: string]: {
    [subpageId: string]: {
      completed: boolean;
      timestamp: Date;
    };
  };
}

const ProgressTracker: React.FC = () => {
  const location = useLocation();
  const { siteConfig } = useDocusaurusContext();
  const [progress, setProgress] = useState<ProgressData>({});
  const [moduleProgress, setModuleProgress] = useState<number>(0);

  // Extract module and subpage from URL
  const pathParts = location.pathname.split('/').filter(part => part);
  const moduleId = pathParts[0] || 'intro';
  
  // Simplified ID extraction for subpages
  const subpageId = pathParts.length > 1 
    ? `${pathParts[0]}/${pathParts.slice(1).join('/')}` 
    : pathParts[0] || 'intro';

  useEffect(() => {
    // Load progress from localStorage
    const savedProgress = localStorage.getItem('textbookProgress');
    if (savedProgress) {
      try {
        const parsedProgress = JSON.parse(savedProgress);
        setProgress(parsedProgress);
        
        // Calculate progress for current module
        if (parsedProgress[moduleId]) {
          const moduleData = parsedProgress[moduleId];
          const completedSubpages = Object.values(moduleData).filter(
            (item: any) => item.completed
          ).length;
          
          const totalSubpages = Object.keys(moduleData).length;
          const progressPercentage = totalSubpages > 0 
            ? Math.round((completedSubpages / totalSubpages) * 100) 
            : 0;
          
          setModuleProgress(progressPercentage);
        }
      } catch (e) {
        console.error('Error loading progress:', e);
      }
    }
  }, [moduleId]);

  useEffect(() => {
    // Mark current page as viewed/started
    if (subpageId) {
      const updatedProgress = {
        ...progress,
        [moduleId]: {
          ...(progress[moduleId] || {}),
          [subpageId]: {
            completed: false, // Marked as completed in handleComplete
            timestamp: new Date()
          }
        }
      };
      setProgress(updatedProgress);
      localStorage.setItem('textbookProgress', JSON.stringify(updatedProgress));
    }
  }, [subpageId, moduleId, progress]);

  const handleComplete = () => {
    if (subpageId && progress[moduleId]?.[subpageId]) {
      const updatedProgress = {
        ...progress,
        [moduleId]: {
          ...progress[moduleId],
          [subpageId]: {
            completed: true,
            timestamp: new Date()
          }
        }
      };
      setProgress(updatedProgress);
      localStorage.setItem('textbookProgress', JSON.stringify(updatedProgress));
      
      // Recalculate module progress
      const moduleData = updatedProgress[moduleId];
      const completedSubpages = Object.values(moduleData).filter(
        (item: any) => item.completed
      ).length;
      
      const totalSubpages = Object.keys(moduleData).length;
      const progressPercentage = totalSubpages > 0 
        ? Math.round((completedSubpages / totalSubpages) * 100) 
        : 0;
      
      setModuleProgress(progressPercentage);
    }
  };

  // Mark as completed when component mounts (for demonstration)
  // In a real implementation, this would happen based on user action
  useEffect(() => {
    handleComplete();
  }, []);

  return (
    <div className="progress-tracker-container">
      <div className="module-progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${moduleProgress}%` }}
        />
        <span className="progress-text">{moduleProgress}% Complete</span>
      </div>
      <div className="progress-actions">
        <button 
          onClick={handleComplete}
          className="mark-complete-button"
          disabled={progress[moduleId]?.[subpageId]?.completed}
        >
          {progress[moduleId]?.[subpageId]?.completed ? 'Completed' : 'Mark Complete'}
        </button>
      </div>
    </div>
  );
};

export default ProgressTracker;