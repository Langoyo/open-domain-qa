import React from 'react'
import './App.css';
import Navbar from './components/Navbar';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom'
import { library } from 'fontawesome'
import { fab } from 'fontawesome'
import Home from './components/pages/Home'
import Footer from './components/Footer'
import Query from './components/pages/Query'
import References from './components/pages/References';
import { Helmet } from 'react-helmet'
import favicon from './favicon.ico';
// import { faCheckSquare, faCoffee } from 'fontawesome'

// library.add(fab, faCheckSquare, faCoffee)
function App() {
  return (
    
    <div className="App">
    
    <Helmet>
        <title>Elise</title>
        <meta name="description" content="Open QA" />
        <meta name="theme-color" content="rgba(34,195,138,1)" />
        <link rel="icon" type="image" href={favicon} sizes="48x48" data-react-helmet="true" />
      </Helmet>
      <Router>
        <Navbar/>
        <Routes>
          <Route path='/' exact element={<Home/>}/>
          <Route path='/query' exact element={<Query/>}/>
          <Route path='/references'exact element={<References/>}/>
        </Routes>
        <Footer/>
      </Router>

    </div>
  );
}

export default App;
