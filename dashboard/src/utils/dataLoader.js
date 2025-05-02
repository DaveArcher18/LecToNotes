/**
 * Utility functions for loading and parsing lecture data
 */

/**
 * Loads transcript data from transcript.json file
 * @returns {Promise<Array>} The parsed transcript data
 */
export const loadTranscriptData = async () => {
  try {
    // Try to fetch from the OUTPUT folder first, then fall back to root location
    let response = await fetch('/OUTPUT/transcript.json');
    
    // If not found in OUTPUT folder, try the root location
    if (!response.ok) {
      response = await fetch('/transcript.json');
    }
    
    if (!response.ok) {
      throw new Error('Failed to load transcript data');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error loading transcript data:', error);
    throw error;
  }
};

/**
 * Loads blackboard data from boards.json file
 * @returns {Promise<Array>} The parsed blackboard data
 */
export const loadBlackboardData = async () => {
  try {
    // Try to fetch from the OUTPUT/boards folder first, then fall back to boards folder
    let response = await fetch('/OUTPUT/boards/boards.json');
    let jsonPath = '/OUTPUT/boards/';
    
    // If not found in OUTPUT/boards folder, try the boards folder
    if (!response.ok) {
      response = await fetch('/boards/boards.json');
      jsonPath = '/boards/';
    }
    
    if (!response.ok) {
      throw new Error('Failed to load blackboard data');
    }
    
    const data = await response.json();
    
    // Process all board paths to ensure they have the correct path prefix
    return data.map(board => ({
      ...board,
      path: `${jsonPath}${board.path}` // Ensure path has the correct prefix
    }));
  } catch (error) {
    console.error('Error loading blackboard data:', error);
    throw error;
  }
};

/**
 * Loads both transcript and blackboard data and returns them
 * @returns {Promise<Object>} Object containing transcript and blackboard data
 */
export const loadLectureData = async () => {
  try {
    const [transcript, blackboards] = await Promise.all([
      loadTranscriptData(),
      loadBlackboardData()
    ]);
    
    return { transcript, blackboards };
  } catch (error) {
    console.error('Error loading lecture data:', error);
    throw error;
  }
};

/**
 * Formats a timestamp from 00:00:00 format to 00_00_00 format or vice versa
 * @param {string} timestamp - The timestamp to format
 * @param {boolean} toUnderscoreFormat - Whether to convert to underscore format
 * @returns {string} The formatted timestamp
 */
export const formatTimestamp = (timestamp, toUnderscoreFormat = false) => {
  if (toUnderscoreFormat) {
    return timestamp.replace(/:/g, '_');
  } else {
    return timestamp.replace(/_/g, ':');
  }
};

/**
 * Converts a timestamp in format 00_00_00 or 00:00:00 to seconds
 * @param {string} timestamp - The timestamp to convert
 * @returns {number} The timestamp in seconds
 */
export const timestampToSeconds = (timestamp) => {
  if (!timestamp) return 0;
  
  // Normalize the timestamp format first
  const normalizedTimestamp = timestamp.replace(/_/g, ':');
  
  // Split the timestamp into hours, minutes, and seconds
  const parts = normalizedTimestamp.split(':');
  
  // Handle potential parsing issues by ensuring we have valid numbers
  const hours = parseInt(parts[0]) || 0;
  const minutes = parseInt(parts[1]) || 0;
  const secondsPart = parts[2] || '0';
  
  // Handle seconds that might include milliseconds (e.g., "12.500")
  let seconds = 0;
  if (secondsPart.includes('.')) {
    const [wholePart, fractionPart] = secondsPart.split('.');
    seconds = parseInt(wholePart) || 0;
    // Add milliseconds precision for more accurate comparisons
    seconds += (parseInt(fractionPart) || 0) / Math.pow(10, fractionPart.length);
  } else {
    seconds = parseInt(secondsPart) || 0;
  }
  
  // Convert to seconds with millisecond precision
  return hours * 3600 + minutes * 60 + seconds;
};

/**
 * Finds the transcript segment that corresponds to the given timestamp
 * @param {Array} transcriptSegments - The transcript segments
 * @param {string} timestamp - The timestamp to find
 * @returns {Object|null} The matching segment or null if not found
 */
export const findTranscriptSegmentByTimestamp = (transcriptSegments, timestamp) => {
  // Normalize the timestamp format
  const normalizedTimestamp = formatTimestamp(timestamp);
  
  return transcriptSegments.find(segment => {
    const segmentStart = formatTimestamp(segment.start);
    const segmentEnd = formatTimestamp(segment.end);
    
    const timestampInSeconds = timestampToSeconds(normalizedTimestamp);
    const startInSeconds = timestampToSeconds(segmentStart);
    const endInSeconds = timestampToSeconds(segmentEnd);
    
    return timestampInSeconds >= startInSeconds && timestampInSeconds < endInSeconds;
  }) || null;
};

/**
 * Finds blackboard images that correspond to the given timestamp or are within a segment
 * @param {Array} blackboards - All blackboard images
 * @param {string} timestamp - The timestamp to find boards for
 * @param {number} timeWindowSeconds - Optional time window in seconds to include boards before and after the timestamp
 * @returns {Array} Array of blackboard image objects
 */
export const findBlackboardsByTimestamp = (blackboards, timestamp, timeWindowSeconds = 60) => {
  // Normalize the timestamp format
  const normalizedTimestamp = formatTimestamp(timestamp);
  const timestampInSeconds = timestampToSeconds(normalizedTimestamp);
  
  return blackboards.filter(board => {
    const boardTimestamp = formatTimestamp(board.timestamp);
    const boardTimeInSeconds = timestampToSeconds(boardTimestamp);
    
    // Include boards that are within the time window of the timestamp
    return Math.abs(boardTimeInSeconds - timestampInSeconds) <= timeWindowSeconds;
  });
};

/**
 * Gets all blackboard images for a specific transcript segment
 * @param {Array} blackboards - All blackboard images
 * @param {Object} segment - The transcript segment
 * @returns {Array} Array of blackboard image objects
 */
export const getBlackboardsForSegment = (blackboards, segment) => {
  if (!segment) return [];
  
  const segmentStart = formatTimestamp(segment.start);
  const segmentEnd = formatTimestamp(segment.end);
  
  const startInSeconds = timestampToSeconds(segmentStart);
  const endInSeconds = timestampToSeconds(segmentEnd);
  
  // Increase the buffer to 2 seconds to better handle timestamp alignment issues
  // This helps with off-by-one errors in timestamp matching that were observed in the UI
  const adjustedStartInSeconds = startInSeconds - 2;
  const adjustedEndInSeconds = endInSeconds + 2;
  
  // Calculate segment duration for more intelligent matching
  const segmentDuration = endInSeconds - startInSeconds;
  
  // Find boards that are within the adjusted time range
  let candidateBoards = blackboards.filter(board => {
    const boardTimestamp = formatTimestamp(board.timestamp);
    const boardTimeInSeconds = timestampToSeconds(boardTimestamp);
    
    // Use the adjusted time range to include boards that might be slightly outside
    // the exact segment boundaries due to rounding or precision issues
    return boardTimeInSeconds >= adjustedStartInSeconds && boardTimeInSeconds <= adjustedEndInSeconds;
  });
  
  // If we found boards within the adjusted range, use them
  if (candidateBoards.length > 0) {
    // For very short segments (less than 5 seconds), be more selective about which boards to include
    if (segmentDuration < 5 && candidateBoards.length > 1) {
      // For short segments, prioritize boards that are closest to the segment midpoint
      const segmentMidpoint = (startInSeconds + endInSeconds) / 2;
      
      // Calculate how close each board is to the segment midpoint
      candidateBoards = candidateBoards.map(board => {
        const boardTimeInSeconds = timestampToSeconds(formatTimestamp(board.timestamp));
        const distanceToMidpoint = Math.abs(boardTimeInSeconds - segmentMidpoint);
        return { ...board, distanceToMidpoint };
      });
      
      // Sort by distance to midpoint and take the closest ones (up to 2 for short segments)
      candidateBoards.sort((a, b) => a.distanceToMidpoint - b.distanceToMidpoint);
      candidateBoards = candidateBoards.slice(0, 2);
      
      // Remove the temporary distance property
      candidateBoards = candidateBoards.map(({ distanceToMidpoint, ...board }) => board);
    }
  } else {
    // If no boards were found within the adjusted range, find the closest board
    const segmentMidpoint = (startInSeconds + endInSeconds) / 2;
    let closestBoard = null;
    let minDistance = Number.MAX_SAFE_INTEGER;
    
    blackboards.forEach(board => {
      const boardTimeInSeconds = timestampToSeconds(formatTimestamp(board.timestamp));
      const distance = Math.abs(boardTimeInSeconds - segmentMidpoint);
      
      // Only consider boards that are within a reasonable time range (30 seconds)
      if (distance < minDistance && distance < 30) {
        minDistance = distance;
        closestBoard = { ...board };
      }
    });
    
    if (closestBoard) {
      candidateBoards = [closestBoard];
    }
  }
  
  // Sort the boards by timestamp to ensure they appear in chronological order
  return candidateBoards.sort((a, b) => {
    return timestampToSeconds(formatTimestamp(a.timestamp)) - 
           timestampToSeconds(formatTimestamp(b.timestamp));
  });
};

/**
 * Groups blackboard images by similar timestamps (within a specified time window)
 * @param {Array} blackboards - Array of blackboard images
 * @param {number} timeWindowSeconds - Time window in seconds to group boards
 * @returns {Array} Array of blackboard groups
 */
/**
 * Determines the color for the timestamp match indicator based on how closely a board matches the segment time
 * @param {string} boardTimestamp - The timestamp of the board
 * @param {string} segmentStart - The start timestamp of the segment
 * @param {string} segmentEnd - The end timestamp of the segment
 * @returns {string} A color value (hex or RGB) representing the match quality
 */
export const getTimestampMatchColor = (boardTimestamp, segmentStart, segmentEnd) => {
  if (!boardTimestamp || !segmentStart || !segmentEnd) {
    return '#cccccc'; // Default gray for missing data
  }
  
  const boardTime = timestampToSeconds(formatTimestamp(boardTimestamp));
  const startTime = timestampToSeconds(formatTimestamp(segmentStart));
  const endTime = timestampToSeconds(formatTimestamp(segmentEnd));
  
  // Calculate segment midpoint
  const midpointTime = (startTime + endTime) / 2;
  
  // Calculate segment duration
  const segmentDuration = endTime - startTime;
  
  // Calculate how far the board is from the segment midpoint, as a percentage of segment duration
  // A value of 0 means it's exactly at the midpoint, 1 means it's at the boundary
  const distanceFromMidpoint = Math.abs(boardTime - midpointTime) / (segmentDuration / 2);
  
  // Cap the distance at 1 (at or beyond segment boundary)
  const normalizedDistance = Math.min(distanceFromMidpoint, 1);
  
  // Color gradient from green (perfect match) to yellow (boundary match) to red (outside segment)
  if (normalizedDistance <= 0.5) {
    // Green to yellow gradient for good matches (0-50% distance)
    const greenComponent = 128 + Math.floor((0.5 - normalizedDistance) * 2 * 127); // 255 to 128
    return `rgb(${greenComponent}, 200, 0)`;
  } else {
    // Yellow to red gradient for boundary matches (50-100% distance)
    const redComponent = 200;
    const greenComponent = Math.floor((1 - normalizedDistance) * 2 * 200); // 200 to 0
    return `rgb(${redComponent}, ${greenComponent}, 0)`;
  }
};

export const groupBlackboardsByTime = (blackboards, timeWindowSeconds = 5) => {
  if (!blackboards || blackboards.length === 0) return [];
  
  // Sort blackboards by timestamp
  const sortedBoards = [...blackboards].sort((a, b) => {
    return timestampToSeconds(formatTimestamp(a.timestamp)) - timestampToSeconds(formatTimestamp(b.timestamp));
  });
  
  const groups = [];
  let currentGroup = [sortedBoards[0]];
  let currentTimestamp = timestampToSeconds(formatTimestamp(sortedBoards[0].timestamp));
  
  for (let i = 1; i < sortedBoards.length; i++) {
    const board = sortedBoards[i];
    const boardTimestamp = timestampToSeconds(formatTimestamp(board.timestamp));
    
    if (boardTimestamp - currentTimestamp <= timeWindowSeconds) {
      // Add to current group if within time window
      currentGroup.push(board);
    } else {
      // Start a new group
      groups.push(currentGroup);
      currentGroup = [board];
      currentTimestamp = boardTimestamp;
    }
  }
  
  // Add the last group
  if (currentGroup.length > 0) {
    groups.push(currentGroup);
  }
  
  return groups;
};