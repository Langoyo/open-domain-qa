import React from 'react'
import Button from './Button'
import {Link} from 'react-router-dom'
import './HomeObject.css' 


function ReferenceObject({
    lightBg, 
    buttonType,
    topLine, 
    lightText, 
    lightTextDesc, 
    headline, 
    description, 
    buttonLabel, 
    img, 
    alt, 
    imageStart,
    buttonLink
}) {
  return (
    <div className={lightBg ? 'home__hero-section' : 'home__hero-section darkBg'}>
        <div className='conatiner'>
            <div className='row home__hero-row'
            style={{display:'flex',flexDirection:imageStart === 'start' ? 'row-reverse' : 'row'}}>
                <div className='col'>
                    <div className = 'home__hero-text-wrapper'>
                        <div className='top-line'>{topLine}</div>
                            <h1 className={lightText?'heading':'heading dark'}>{headline}</h1>
                            <p className={lightTextDesc ? 'home__hero-subtitle light' : 'home__hero_subtitle dark'}>{description}</p>
                            
                                <a style={{textDecoration:'none'}} href={buttonLink} target="_blank" className={'btn--dark'}>Source</a>
                            



                        </div>
                    
                </div>
                <div className='col'>
                    <div className='home__hero-img-wrapper'></div>
                    <img src={img} alt={alt} className = 'home__hero-img'/> 
                </div>
            </div>
        </div>
    </div>
  )
};

export default ReferenceObject