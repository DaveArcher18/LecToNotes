import React, { useState, useEffect } from 'react';
import { loadLectureData, formatTimestamp, findTranscriptSegmentByTimestamp, getBlackboardsForSegment } from './utils/dataLoader';
import TranscriptViewer from './components/TranscriptViewer';
import BlackboardViewer from './components/BlackboardViewer';
import './App.css';

function App() {
  const [lectureData, setLectureData] = useState(null);
  const [currentTime, setCurrentTime] = useState(null);
  const [currentSegment, setCurrentSegment] = useState(null);
  const [currentBoards, setCurrentBoards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load lecture data on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await loadLectureData();
        setLectureData(data);
        
        // Set initial time to the first segment
        if (data.transcript && data.transcript.length > 0) {
          const initialTime = formatTimestamp(data.transcript[0].start);
          setCurrentTime(initialTime);
          setCurrentSegment(data.transcript[0]);
          setCurrentBoards(getBlackboardsForSegment(data.blackboards, data.transcript[0]));
        }
        
        setLoading(false);
      } catch (err) {
        setError('Failed to load lecture data. Please try again later.');
        setLoading(false);
        console.error(err);
      }
    };
    
    fetchData();
  }, []);

  // Handle time change (when clicking on transcript or blackboard)
  const handleTimeChange = (newTime) => {
    setCurrentTime(newTime);
    
    if (lectureData) {
      const segment = findTranscriptSegmentByTimestamp(lectureData.transcript, newTime);
      if (segment) {
        setCurrentSegment(segment);
        setCurrentBoards(getBlackboardsForSegment(lectureData.blackboards, segment));
      }
    }
  };

  if (loading) {
    return <div className="loading">Loading lecture content...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Lecture Content Dashboard</h1>
      </header>
      
      <main className="app-content">
        {lectureData ? (
          <>
            <TranscriptViewer 
              segments={lectureData.transcript} 
              currentTime={currentTime}
              currentSegment={currentSegment}
              onTimeChange={handleTimeChange}
            />
            
            <BlackboardViewer 
              boards={lectureData.blackboards} 
              currentTime={currentTime}
              currentSegment={currentSegment}
              onTimeChange={handleTimeChange}
            />
          </>
        ) : (
          <div className="no-data">No lecture data available.</div>
        )}
      </main>
    </div>
  );
}

export default App;
