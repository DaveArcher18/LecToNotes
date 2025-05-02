import React, { useRef, useEffect, useState } from 'react';
import { formatTimestamp } from '../utils/dataLoader';
import './TranscriptViewer.css';

const TranscriptViewer = ({ segments, currentTime, currentSegment, onTimeChange }) => {
  const segmentsRef = useRef(null);
  const activeSegmentRef = useRef(null);
  const [viewMode, setViewMode] = useState('transcript'); // 'transcript' or 'summary'

  // Handle clicking on a transcript segment
  const handleSegmentClick = (segmentStartTime) => {
    onTimeChange(formatTimestamp(segmentStartTime));
  };

  // Scroll to active segment when it changes
  useEffect(() => {
    if (activeSegmentRef.current && segmentsRef.current) {
      activeSegmentRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }, [currentSegment]);

  // Toggle between transcript and summary view
  const toggleViewMode = () => {
    setViewMode(viewMode === 'transcript' ? 'summary' : 'transcript');
  };

  return (
    <div className="transcript-viewer">
      <div className="viewer-header">
        <h2>Lecture {viewMode === 'transcript' ? 'Transcript' : 'Summary'}</h2>
        <button 
          className="toggle-view-button"
          onClick={toggleViewMode}
        >
          Show {viewMode === 'transcript' ? 'Summary' : 'Transcript'}
        </button>
      </div>
      <div className="transcript-segments" ref={segmentsRef}>
        {segments.map((segment, index) => (
          <div 
            key={index}
            className={`transcript-segment ${segment.start === currentSegment?.start ? 'active' : ''}`}
            onClick={() => handleSegmentClick(segment.start)}
            ref={segment.start === currentSegment?.start ? activeSegmentRef : null}
          >
            <div className="segment-time">
              {formatTimestamp(segment.start)} - {formatTimestamp(segment.end)}
            </div>
            {viewMode === 'transcript' ? (
              <div className="segment-content">
                {segment.content || 'No transcript available'}
              </div>
            ) : (
              <div className="segment-summary">
                {segment.summary ? (
                  <div>
                    <div className="summary-content">{segment.summary}</div>
                  </div>
                ) : (
                  <div className="no-summary">No summary available</div>
                )}
              </div>
            )}
            <button 
              className="copy-button" 
              onClick={(e) => {
                e.stopPropagation();
                const textToCopy = viewMode === 'transcript' ? 
                  (segment.content || '') : 
                  (segment.summary || '');
                navigator.clipboard.writeText(textToCopy);
                alert(`${viewMode === 'transcript' ? 'Transcript' : 'Summary'} copied to clipboard!`);
              }}
            >
              Copy Text
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TranscriptViewer;