import React, { useRef, useEffect } from 'react';
import { formatTimestamp } from '../utils/dataLoader';
import './TranscriptViewer.css';

const TranscriptViewer = ({ segments, currentTime, currentSegment, onTimeChange }) => {
  const segmentsRef = useRef(null);
  const activeSegmentRef = useRef(null);

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

  return (
    <div className="transcript-viewer">
      <h2>Lecture Transcript</h2>
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
            <div className="segment-content">
              {segment.content || 'No transcript available'}
            </div>
            <button 
              className="copy-button" 
              onClick={(e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(segment.content || '');
                alert('Transcript copied to clipboard!');
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