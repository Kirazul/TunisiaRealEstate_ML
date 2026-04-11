# Frontend - Tunisia Real Estate AI

    frontend for the Tunisia Real Estate Price Prediction system.

## 📁 Structure

```
frontend/
├── assets/
│   ├── icons/          # Application icons
│   │   └── favicon.svg
│   └── data/           # Static data files
│       ├── atlas.geojson
│       ├── TN-gouvernorats.geojson
│       ├── zone_coverage.json
│       ├── delegation_profiles.json
│       └── delegations.geojson
├── css/
│   └── style.css       # Main stylesheet
├── js/
│   ├── map.js          # Interactive map functionality
│   └── notebook.js     # ML notebook interface
├── index.html          # Main landing page with map
└── notebook.html       # ML Lab notebook interface
```

## 🚀 Pages

### 1. **index.html** - Main Landing Page
- Interactive map of Tunisia with delegation coverage
- Real-time price predictions
- Property search interface
- Model performance metrics

### 2. **notebook.html** - ML Lab
- Complete pipeline documentation
- 56 executable code cells across 8 chapters
- Live code execution with Python kernel
- Q&A section for oral defense

## 🎨 Features

- **Glass Morphism Design** - Modern, elegant UI
- **Responsive Layout** - Works on all devices
- **Dark Theme** - Eye-friendly interface
- **Interactive Visualizations** - D3.js and Chart.js
- **Code Editor** - CodeMirror with syntax highlighting
- **Real-time Updates** - Live data from backend API

## 🔧 Technologies

- **HTML5** - Semantic markup
- **CSS3** - Modern styling with animations
- **Vanilla JavaScript** - No framework dependencies
- **D3.js** - Geographic visualizations
- **Chart.js** - Statistical charts
- **CodeMirror** - Code editor
- **FastAPI Backend** - Python API server

## 📊 Data Files

All data files are organized in `assets/data/`:

- **atlas.geojson** - Tunisia delegation boundaries
- **TN-gouvernorats.geojson** - Governorate boundaries
- **zone_coverage.json** - Model coverage data
- **delegation_profiles.json** - Historical price profiles
- **delegations.geojson** - Detailed delegation data

## 🌐 API Endpoints

The frontend connects to these backend endpoints:

- `GET /` - Serves index.html
- `GET /notebook` - Serves notebook.html
- `GET /model_summary` - Model metrics
- `POST /predict` - Price predictions
- `GET /delegations` - Available delegations
- `GET /health` - Health check

## 🎯 Production Checklist

- ✅ Organized folder structure
- ✅ Minified and versioned assets
- ✅ No unused files
- ✅ Proper error handling
- ✅ Cache control headers
- ✅ CORS configured
- ✅ Mobile responsive
- ✅ Accessibility compliant

## 📝 Notes

- All paths are relative for easy deployment
- Version numbers in URLs for cache busting
- CDN resources for external libraries
- No build step required - pure static files
