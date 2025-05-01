# LecToNotes Dashboard

This dashboard provides a web interface for viewing lecture transcripts and blackboard content.

## Setup Instructions

### Setting up the OUTPUT folder

The dashboard is configured to look for lecture data in the `OUTPUT` folder. Follow these steps to set up the data correctly:

1. Create an `OUTPUT` folder in the public directory of the dashboard:

```bash
mkdir -p public/OUTPUT
mkdir -p public/OUTPUT/boards
```

2. Copy the transcript.json file to the OUTPUT folder:

```bash
cp ../transcript.json public/OUTPUT/
```

3. Copy the boards.json file to the OUTPUT/boards folder:

```bash
cp ../boards/boards.json public/OUTPUT/boards/
```

4. If you have board images, copy them to the appropriate location in the OUTPUT folder.

### Running the Dashboard

1. Install dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Open the dashboard in your browser at http://localhost:5173

## Data Structure

The dashboard expects the following data structure:

- `OUTPUT/transcript.json`: Contains the transcript data
- `OUTPUT/boards/boards.json`: Contains the blackboard data
- Board images should be referenced in the boards.json file with paths relative to the OUTPUT folder

## Automatic Fallback

If files are not found in the OUTPUT folder, the dashboard will automatically try to load them from their original locations:

- `/transcript.json`
- `/boards/boards.json`

This ensures backward compatibility with the existing setup.

## Technical Details

This dashboard is built with React and Vite, providing a fast development experience with HMR (Hot Module Replacement).

The following plugins are used:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
