import React, { Component }  from 'react';
import common_es from "./translations/es/common.json";
import common_en from "./translations/en/common.json";
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import {initializeApp} from 'firebase/app'
import { getAnalytics } from "firebase/analytics";
import {I18nextProvider} from "react-i18next";
import i18next from "i18next";



const firebaseConfig = {

  apiKey: "AIzaSyA9Q2SWgJk-xbpnhgWfTCYmG4qOr7x_AVE",

  authDomain: "open-qa-cb9f9.firebaseapp.com",

  projectId: "open-qa-cb9f9",

  storageBucket: "open-qa-cb9f9.appspot.com",

  messagingSenderId: "603238817256",

  appId: "1:603238817256:web:c2ad5d98c6de4348d4185d",

  measurementId: "G-5TRVS7HSF7"

};


// Initialize Firebase

const app = initializeApp(firebaseConfig);

const analytics = getAnalytics(app);

const root = ReactDOM.createRoot(document.getElementById('root'));


i18next.init({
  interpolation: { escapeValue: false },  // React already does escaping
  lng: 'en',                              // language to use
  resources: {
      en: {
          common: common_en               // 'common' is our custom namespace
      },
      es: {
          common: common_es
      },
  },
});

root.render(
  
  <React.StrictMode>
      <I18nextProvider i18n={i18next}>
        <App />
      </I18nextProvider>
  </React.StrictMode>
);



// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
