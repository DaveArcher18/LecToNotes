import React, { useState, useRef, useEffect } from 'react';
import { formatTimestamp, getBlackboardsForSegment, getTimestampMatchColor } from '../utils/dataLoader';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './BlackboardViewer.css';

// Function to preprocess LaTeX content before rendering
const preprocessLatex = (text) => {
  if (!text) return '';
  
  // Fix common LaTeX errors
  const processedText = text
    // Fix double dollar signs with newlines between them
    .replace(/\$\$\s*\n+\s*\$\$/g, '\n\n')
    
    // Fix escaped backslashes that shouldn't be escaped
    .replace(/\\\\([a-zA-Z]+)/g, '\\$1')
    
    // Fix double backslashes in LaTeX commands
    .replace(/\\\\begin/g, '\\begin')
    .replace(/\\\\end/g, '\\end')
    .replace(/\\\\text/g, '\\text')
    .replace(/\\\\mathbb/g, '\\mathbb')
    .replace(/\\\\section/g, '\\section')
    .replace(/\\\\subsection/g, '\\subsection')
    
    // Fix common OCR errors with numbers in commands
    .replace(/\\(\d+)/g, '{$1}')
    .replace(/\\_/g, '_')
    .replace(/\\u0007/g, '')
    .replace(/\\u0007pprox/g, '\\approx')
    .replace(/\\_\{(\d+)\}/g, '_{$1}')
    .replace(/\\\^\{(\d+)\}/g, '^{$1}')
    
    // Remove any LaTeX document commands
    .replace(/\\documentclass(\{.*?\})/g, '')
    .replace(/\\usepackage(\{.*?\})/g, '')
    .replace(/\\begin\{document\}/g, '')
    .replace(/\\end\{document\}/g, '')
    
    // Fix mismatched math delimiters
    .replace(/\$\$(.*?)\$/g, '$$$$1$$')
    .replace(/\$(.*?)\$\$/g, '$$$1$')
    
    // Fix common environment issues
    .replace(/\\begin\{array\}(\s*)\n/g, '\\begin{array}$1')
    .replace(/\\end\{array\}(\s*)\n/g, '\\end{array}$1')
    
    // Fix common bracket issues
    .replace(/\\left\(\s*\\right\)/g, '()')
    .replace(/\\left\[\s*\\right\]/g, '[]')
    .replace(/\\left\\{\s*\\right\\}/g, '{}')
    
    // Fix common operators
    .replace(/\\operatorname\{([^}]+)\}/g, '\\text{$1}')
    
    // Fix incomplete environments
    .replace(/\\begin\{([^}]+)\}(?![\s\S]*?\\end\{\1\})/g, (match, env) => `${match}\n\\end{${env}}`)
    
    // Fix common fraction errors
    .replace(/\\frac([^{])/g, '\\frac{$1}')
    .replace(/\\frac\{([^}]+)\}([^{])/g, '\\frac{$1}{$2}')
    
    // Replace malformed LaTeX with properly formed equivalents
    .replace(/\\text\s+\{/g, '\\text{')
    .replace(/\\text\{([^}]*)\}/g, '\\text{$1}')
    
    // Fix incorrect equation environments 
    .replace(/\$\$ \\begin\{align/g, '\\begin{align')
    .replace(/\\end\{align\} \$\$/g, '\\end{align}');
  
  return processedText;
};

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
                      src={board.path} 
                      alt={`Blackboard at ${formatTimestamp(board.timestamp)}`} 
                      className="board-image"
                      onError={(e) => {
                        console.error(`Failed to load image: ${board.path}`);
                        // Try alternative path formats if the original fails
                        const fileName = board.path.split('/').pop();
                        e.target.src = `/boards/${fileName}`;
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
                        throwOnError: false, // Don't throw on parse errors
                        errorColor: '#cc0000', // Red color for errors
                        maxSize: 100, // Increase size limit for expressions
                        maxExpand: 1000, // Allow more macro expansions
                        macros: {
                          // Common math macros used in lectures
                          "\\R": "\\mathbb{R}",
                          "\\N": "\\mathbb{N}",
                          "\\Z": "\\mathbb{Z}",
                          "\\C": "\\mathbb{C}",
                          "\\Q": "\\mathbb{Q}",
                          "\\F": "\\mathbb{F}",
                          "\\implies": "\\Rightarrow",
                          "\\iff": "\\Leftrightarrow",
                          "\\to": "\\rightarrow",
                          "\\mapsto": "\\rightarrow",
                          "\\xrightarrow": "\\rightarrow",
                          
                          // Fix common LaTeX issues
                          "\\2": "{2}",
                          "\\3": "{3}",
                          "\\4": "{4}",
                          "\\5": "{5}",
                          "\\u0007": "",
                          "\\u0007pprox": "\\approx",
                          "\\u0007cute{e}t": "\\acute{e}t",
                          
                          // Common text operators
                          "\\section": "\\section*",
                          "\\subsection": "\\subsection*",
                          "\\mid": " ",
                          "\\Shirai": "\\text{Shirai}",
                          "\\Showai": "\\text{Showai}",
                          "\\So": "\\text{So}",
                          
                          // Common mathematical operators
                          "\\operatorname{sing}": "\\text{sing}",
                          "\\operatorname{dR}": "\\text{dR}",
                          "\\operatorname{et}": "\\text{\u00e9t}",
                          "\\operatorname{crys}": "\\text{crys}",
                          "\\dR": "\\text{dR}",
                          "\\et": "\\text{\u00e9t}",
                          "\\crys": "\\text{crys}",
                          "\\Spec": "\\text{Spec}",
                          "\\Hom": "\\text{Hom}",
                          "\\End": "\\text{End}",
                          "\\colim": "\\text{colim}",
                          "\\lim": "\\text{lim}",
                          "\\coker": "\\text{coker}"
                        }
                      }]]}
                      components={{
                        // Custom components to handle LaTeX-related issues
                        p: ({node, ...props}) => {
                          // Check if this paragraph contains only LaTeX and nothing else
                          const containsOnlyMath = 
                            node.children.length === 1 && 
                            node.children[0].type === 'element' && 
                            (node.children[0].properties?.className?.includes('math-display') || 
                             node.children[0].properties?.className?.includes('math-inline'));
                          
                          return containsOnlyMath ? <>{props.children}</> : <p {...props} />;
                        },
                        // Better handle code blocks that might be mistaken for LaTeX
                        code: ({node, inline, ...props}) => {
                          // Check if this might be intended as LaTeX
                          if (!inline && props.children && typeof props.children === 'string' && 
                              (props.children.includes('\\begin') || props.children.includes('\\frac'))) {
                            // Convert to math display
                            return <div className="math math-display">{'$$' + props.children + '$$'}</div>;
                          }
                          return inline ? <code {...props} /> : <pre><code {...props} /></pre>;
                        }
                      }}
                    >
                      {preprocessLatex(board.text || '\n\n*No markdown content available for this blackboard image.*')}
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
                      src={board.path} 
                      alt={`Blackboard at ${formatTimestamp(board.timestamp)}`} 
                      className="board-image"
                      onError={(e) => {
                        console.error(`Failed to load image: ${board.path}`);
                        // Try alternative path formats if the original fails
                        const fileName = board.path.split('/').pop();
                        e.target.src = `/boards/${fileName}`;
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
                        throwOnError: false,
                        errorColor: '#cc0000',
                        maxSize: 100,
                        maxExpand: 1000,
                        macros: {
                          "\\R": "\\mathbb{R}",
                          "\\N": "\\mathbb{N}",
                          "\\Z": "\\mathbb{Z}",
                          "\\C": "\\mathbb{C}",
                          "\\Q": "\\mathbb{Q}",
                          "\\F": "\\mathbb{F}",
                          "\\implies": "\\Rightarrow",
                          "\\iff": "\\Leftrightarrow",
                          "\\to": "\\rightarrow",
                          "\\mapsto": "\\rightarrow",
                          "\\xrightarrow": "\\rightarrow",
                          "\\2": "{2}",
                          "\\3": "{3}",
                          "\\4": "{4}",
                          "\\5": "{5}",
                          "\\u0007": "",
                          "\\u0007pprox": "\\approx",
                          "\\u0007cute{e}t": "\\acute{e}t",
                          "\\section": "\\section*",
                          "\\subsection": "\\subsection*",
                          "\\mid": " ",
                          "\\Shirai": "\\text{Shirai}",
                          "\\Showai": "\\text{Showai}",
                          "\\So": "\\text{So}",
                          "\\operatorname{sing}": "\\text{sing}",
                          "\\operatorname{dR}": "\\text{dR}",
                          "\\operatorname{et}": "\\text{\u00e9t}",
                          "\\operatorname{crys}": "\\text{crys}",
                          "\\dR": "\\text{dR}",
                          "\\et": "\\text{\u00e9t}",
                          "\\crys": "\\text{crys}",
                          "\\Spec": "\\text{Spec}",
                          "\\Hom": "\\text{Hom}",
                          "\\End": "\\text{End}",
                          "\\colim": "\\text{colim}",
                          "\\lim": "\\text{lim}",
                          "\\coker": "\\text{coker}"
                        }
                      }]]}
                      components={{
                        p: ({node, ...props}) => {
                          const containsOnlyMath = 
                            node.children.length === 1 && 
                            node.children[0].type === 'element' && 
                            (node.children[0].properties?.className?.includes('math-display') || 
                             node.children[0].properties?.className?.includes('math-inline'));
                          
                          return containsOnlyMath ? <>{props.children}</> : <p {...props} />;
                        },
                        code: ({node, inline, ...props}) => {
                          if (!inline && props.children && typeof props.children === 'string' && 
                              (props.children.includes('\\begin') || props.children.includes('\\frac'))) {
                            return <div className="math math-display">{'$$' + props.children + '$$'}</div>;
                          }
                          return inline ? <code {...props} /> : <pre><code {...props} /></pre>;
                        }
                      }}
                    >
                      {preprocessLatex(board.text || '\n\n*No markdown content available for this blackboard image.*')}
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