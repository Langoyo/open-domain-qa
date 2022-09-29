import React from 'react'
import '../../App.css'
import './Home.css'
import Footer from '../Footer'
import HomeObject from '../HomeObject'
import Button from '../Button'
import {Link} from 'react-router-dom'
import { useRef } from 'react'
import query from '../../images/img-query.jpg';
import chat_img from '../../images/chat.jpg';
import table_img from '../../images/table.jpg';
import advanced_img from '../../images/advanced.jpg';
import language_img from '../../images/language.jpg';
import wiki_img from '../../images/wiki.jpg';
import {useTranslation} from "react-i18next";
import {DirectLink, Element, Events, animateScroll as scroll, scrollSpy, scroller } from 'react-scroll'


function Home(){
const {t, i18n} = useTranslation('common');
const scollToRef = useRef()


const searcAnything = {
    bgColor: 'whiteBg',
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: t('tuto.search.headline'),
    description: t('tuto.search.description'),
    img:query,
    imageStart:'row',
    alt:'/images/query-img.jpg',
    buttonLabel:t('tuto.search.button'),
    buttonLink:'/query'
}


const chat = {
    bgColor: 'greenBg',
    lightText: true,
    buttonType:'btn--outline',
    lightTextDesc: true,
    topLine: '',
    headline: t('tuto.chat.headline'),
    description:t('tuto.chat.description'), 
    buttonLable:'Go',
    img:chat_img,
    imageStart:'start',
    alt:'/images/chat.jpg',
    buttonLabel:t('tuto.chat.button'),
    buttonLink:'/query'
}


const advanced = {
    bgColor: 'whiteBg',
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: t('tuto.advanced.headline'),
    description:t('tuto.advanced.description'), 
    img:advanced_img,
    imageStart:'row',
    alt:'/images/advanced.jpg',
    buttonLabel:t('tuto.search.button'),
    buttonLink:'/query'
}

const table = {
    bgColor: 'greenBg',
    lightText: true,
    buttonType:'btn--outline',
    lightTextDesc: true,
    topLine: '',
    headline: t('tuto.table.headline'),
    description: t('tuto.table.description'),
    img:table_img,
    imageStart:'start',
    alt:'/images/table.jpg',
    buttonLabel:t('tuto.table.button'),
    buttonLink:'/query'
}


const language = {
    bgColor: 'whiteBg',
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: t('tuto.language.headline'),
    description: t('tuto.language.description'),
    buttonLable:'Try',
    img:language_img,
    imageStart:'row',
    alt:'/images/advanced.png',
    buttonLabel:t('tuto.language.button'),
    buttonLink:'/query'
}


const refe = {
    bgColor: 'greenBg',
    lightText: true,
    buttonType:'btn--outline',
    lightTextDesc: true,
    topLine: '',
    headline: t('tuto.references.headline'),
    description: t('tuto.references.description'),
    img:wiki_img,
    imageStart:'start',
    alt:'/images/table.jpg',
    buttonLabel:t('tuto.references.button'),
    buttonLink:'/references'
}




    return (
        <div>
            <div className = 'hero-container'>
        <h2>ELISE</h2>
        <p>{t('home.QA')}</p>
            <div className='hero-btns'>
                 <Button  link='/query'classnmae='bins' buttonStyle='btn--outline' buttonSize='btn--large'>
                 <Link className='links' to='/query'>{t('home.button1')}</Link>       
                 </Button>
                 
                  <Button className='button-scroll'link='/'classnmae='bins' buttonStyle='btn--primary' buttonSize='btn--large' onClick={() => document.querySelector("body").scrollTo({
                        top: 1000,
                        behavior: 'smooth',
                        })}> 
                 {t('home.button2')} 
                 </Button>
            
            </div> 
    </div>
            <HomeObject ref={scollToRef} className="tuto"{...searcAnything}></HomeObject>
            <HomeObject {...chat}></HomeObject>
            <HomeObject {...advanced}></HomeObject>
            <HomeObject {...table}></HomeObject>
            <HomeObject {...language}></HomeObject>
            <HomeObject {...refe}></HomeObject>

        </div>

    )
}

export default Home