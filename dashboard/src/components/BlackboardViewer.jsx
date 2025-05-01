import React, { useState, useRef, useEffect } from 'react';
import { formatTimestamp, getBlackboardsForSegment, getTimestampMatchColor } from '../utils/dataLoader';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './BlackboardViewer.css';

const BlackboardViewer = ({ boards, currentTime, onTimeChange, currentSegment }) => {
  const [activeGroupIndex, setActiveGroupIndex] = useState(0);
  const boardsContainerRef = useRef(null);
  const [segmentBoards, setSegmentBoards] = useState([]);
  const groupedBoards = groupBoardsByTimestamp(boards);

  // Function to group boards by timestamp (within a small time window)
  function groupBoardsByTimestamp(boards) {
    if (!boards || boards.length === 0) return [];
    
    // First, ensure boards are sorted by timestamp
    const sortedBoards = [...boards].sort((a, b) => {
      const aTime = new Date(`2000-01-01T${formatTimestamp(a.timestamp)}`).getTime();
      const bTime = new Date(`2000-01-01T${formatTimestamp(b.timestamp)}`).getTime();
      return aTime - bTime;
    });
    
    const groups = [];
    let currentGroup = [sortedBoards[0]];
    let currentTimestamp = formatTimestamp(sortedBoards[0].timestamp);
    
    for (let i = 1; i < sortedBoards.length; i++) {
      const boardTimestamp = formatTimestamp(sortedBoards[i].timestamp);
      
      // Use a more precise time difference calculation
      const currentTime = new Date(`2000-01-01T${currentTimestamp}`);
      const boardTime = new Date(`2000-01-01T${boardTimestamp}`);
      
      // Calculate time difference in seconds with millisecond precision
      const timeDiff = Math.abs(boardTime - currentTime) / 1000;
      
      // Use a slightly more flexible threshold (3 seconds) to better group related boards
      if (timeDiff <= 3) {
        currentGroup.push(sortedBoards[i]);
      } else {
        groups.push(currentGroup);
        currentGroup = [sortedBoards[i]];
        currentTimestamp = boardTimestamp;
      }
    }
    
    if (currentGroup.length > 0) {
      groups.push(currentGroup);
    }
    
    return groups;
  }

  // Handle scrolling to the next or previous group
  const handleScroll = (direction) => {
    if (direction === 'left' && activeGroupIndex > 0) {
      setActiveGroupIndex(activeGroupIndex - 1);
    } else if (direction === 'right' && activeGroupIndex < groupedBoards.length - 1) {
      setActiveGroupIndex(activeGroupIndex + 1);
    }
  };

  // Copy markdown content to clipboard
  const copyMarkdown = (text) => {
    navigator.clipboard.writeText(text);
    alert('Markdown copied to clipboard!');
  };

  useEffect(() => {
    // Find the group that corresponds to the current time
    if (currentTime && boards.length > 0 && groupedBoards.length > 0) {
      // Convert current time to seconds with millisecond precision
      const currentTimeDate = new Date(`2000-01-01T${currentTime}`);
      const currentTimeSeconds = currentTimeDate.getTime() / 1000;
      
      // Find the closest group by timestamp with preference for boards that appear slightly before
      // the current time rather than after (to match the natural flow of lecture content)
      let closestIndex = 0;
      let minTimeDiff = Number.MAX_SAFE_INTEGER;
      
      for (let i = 0; i < groupedBoards.length; i++) {
        const groupTimestamp = formatTimestamp(groupedBoards[i][0].timestamp);
        const groupTimeDate = new Date(`2000-01-01T${groupTimestamp}`);
        const groupTimeSeconds = groupTimeDate.getTime() / 1000;
        
        // Calculate time difference in seconds
        const timeDiff = currentTimeSeconds - groupTimeSeconds;
        
        // Improved weighting algorithm:
        // - Boards that appear before current time (positive timeDiff) are preferred
        // - Boards that appear after current time (negative timeDiff) are penalized slightly
        // - The closer to the current time, the better
        const adjustedTimeDiff = timeDiff >= 0 
          ? timeDiff // Boards before current time - use actual difference
          : Math.abs(timeDiff) * 1.2; // Boards after current time - penalize by 20%
        
        if (adjustedTimeDiff < minTimeDiff) {
          minTimeDiff = adjustedTimeDiff;
          closestIndex = i;
        }
      }
      
      // Only update if we found a reasonably close match or if this is the initial load
      if (minTimeDiff < 300) { // Within 5 minutes
        setActiveGroupIndex(closestIndex);
      }
    }
  }, [currentTime, boards, groupedBoards]);

  
  // Update segment boards when current segment changes
  useEffect(() => {
    if (currentSegment && boards.length > 0) {
      // Get all boards that fall within the current segment's time range
      // The improved getBlackboardsForSegment function now handles timestamp alignment issues
      const boardsInSegment = getBlackboardsForSegment(boards, currentSegment);
      
      // Set the segment boards directly - the improved function already handles edge cases
      setSegmentBoards(boardsInSegment);
      
      // Log information for debugging
      if (boardsInSegment.length > 0) {
        console.log(`Found ${boardsInSegment.length} boards for segment ${formatTimestamp(currentSegment.start)} - ${formatTimestamp(currentSegment.end)}`);
      } else {
        console.log(`No boards found for segment ${formatTimestamp(currentSegment.start)} - ${formatTimestamp(currentSegment.end)}`);
      }
    } else {
      setSegmentBoards([]);
    }
  }, [currentSegment, boards]);


  return (
    <div className="blackboard-viewer">
      <h2>Blackboard Content</h2>
      
      <div className="blackboard-content">
        {currentSegment && segmentBoards.length > 0 ? (
          <>
            <div className="boards-container" ref={boardsContainerRef}>
              {segmentBoards.map((board, index) => (
                <div key={index} className="board-item">
                  <div className="board-timestamp">
                    {formatTimestamp(board.timestamp)}
                    {/* Visual indicator showing how close this board is to the segment time */}
                    {currentSegment && (
                      <span className="timestamp-match-indicator" 
                            title="Shows how closely this board matches the current segment time"
                            style={{
                              backgroundColor: getTimestampMatchColor(
                                board.timestamp, 
                                currentSegment.start, 
                                currentSegment.end
                              )
                            }}>
                      </span>
                    )}
                  </div>
                  <div className="board-image-container">
                    <img 
                      src={board.path.startsWith('/') ? board.path : `/${board.path}`} 
                      alt={`Blackboard at ${formatTimestamp(board.timestamp)}`} 
                      className="board-image"
                      onError={(e) => {
                        console.error(`Failed to load image: ${board.path}`);
                        // Try alternative path formats if the original fails
                        if (!e.target.src.includes('/boards/')) {
                          e.target.src = `/boards/${board.path.split('/').pop()}`;
                        }
                      }}
                    />
                  </div>
                  <div className="board-markdown">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[[rehypeKatex, {
                        // Enhanced KaTeX options for better rendering
                        strict: false, // Don't throw errors for unsupported commands
                        output: 'html', // Use HTML output for better quality
                        trust: true, // Allow potentially dangerous commands (only use with trusted content)
                        macros: {
                          // Common math macros used in lectures
                          "\\R": "\\mathbb{R}",
                          "\\N": "\\mathbb{N}",
                          "\\Z": "\\mathbb{Z}",
                          "\\C": "\\mathbb{C}",
                          "\\Q": "\\mathbb{Q}",
                          "\\implies": "\\Rightarrow",
                          "\\iff": "\\Leftrightarrow",
                          "\\degree": "^{\\circ}"
                        },
                        displayMode: false, // Let KaTeX determine display mode from delimiters
                        throwOnError: false, // Don't throw on parse errors
                        errorColor: '#cc0000', // Red color for errors
                        minRuleThickness: 0.05, // Improve thin lines rendering
                        fleqn: false, // Display math flush left
                        leqno: false, // Place equation numbers on the left
                        colorIsTextColor: true, // Make \color work like LaTeX
                        maxSize: 10, // Maximum size for font metrics
                        maxExpand: 1000 // Maximum number of macro expansions
                      }]]
                    }
                    >
                      {board.text || '\n\n*No markdown content available for this blackboard image.*'}
                    </ReactMarkdown>
                    <button 
                      className="copy-markdown-button" 
                      onClick={() => copyMarkdown(board.text || '')}
                    >
                      Copy Markdown
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Scroll controls and pagination removed as requested */}
          </>
        ) : groupedBoards.length > 0 ? (
          <>
            <div className="boards-container" ref={boardsContainerRef}>
              {groupedBoards[activeGroupIndex].map((board, index) => (
                <div key={index} className="board-item">
                  <div className="board-timestamp">
                    {formatTimestamp(board.timestamp)}
                    {/* Visual indicator showing how close this board is to the segment time */}
                    {currentSegment && (
                      <span className="timestamp-match-indicator" 
                            title="Shows how closely this board matches the current segment time"
                            style={{
                              backgroundColor: getTimestampMatchColor(
                                board.timestamp, 
                                currentSegment.start, 
                                currentSegment.end
                              )
                            }}>
                      </span>
                    )}
                  </div>
                  <div className="board-image-container">
                    <img 
                      src={board.path.startsWith('/') ? board.path : `/${board.path}`} 
                      alt={`Blackboard at ${formatTimestamp(board.timestamp)}`} 
                      className="board-image"
                      onError={(e) => {
                        console.error(`Failed to load image: ${board.path}`);
                        // Try alternative path formats if the original fails
                        if (!e.target.src.includes('/boards/')) {
                          e.target.src = `/boards/${board.path.split('/').pop()}`;
                        }
                      }}
                    />
                  </div>
                  <div className="board-markdown">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[[rehypeKatex, {
                        // Enhanced KaTeX options for better rendering
                        strict: false,
                        output: 'html',
                        trust: true,
                        macros: {
                          "\\R": "\\mathbb{R}",
                          "\\N": "\\mathbb{N}",
                          "\\Z": "\\mathbb{Z}",
                          "\\C": "\\mathbb{C}",
                          "\\Q": "\\mathbb{Q}",
                          "\\implies": "\\Rightarrow",
                          "\\iff": "\\Leftrightarrow",
                          "\\degree": "^{\\circ}"
                        },
                        displayMode: false,
                        throwOnError: false,
                        errorColor: '#cc0000',
                        minRuleThickness: 0.05,
                        fleqn: false,
                        leqno: false,
                        colorIsTextColor: true,
                        maxSize: 10,
                        maxExpand: 1000
                      }]]
                    }
                    >
                      {board.text || '\n\n*No markdown content available for this blackboard image.*'}
                    </ReactMarkdown>
                    <button 
                      className="copy-markdown-button" 
                      onClick={() => copyMarkdown(board.text || '')}
                    >
                      Copy Markdown
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="no-boards">No blackboard content available for this time period.</div>
        )}
      </div>
    </div>
  );
};

export default BlackboardViewer;