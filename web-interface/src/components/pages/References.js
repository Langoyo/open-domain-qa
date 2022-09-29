import React from 'react'
import ReferenceObject from "../ReferenceObject"
import wiki_img from '../../images/wiki.jpg';
import tfidf_img from '../../images/tfidf.jpg';
import bm25_img from '../../images/bm25.jpg';
import w2v_img from '../../images/w2v.jpg';
import st_img from '../../images/st.jpg';
import bert_img from '../../images/bert.jpg';
import squad_img from '../../images/squad.jpg';
import drqa_img from '../../images/drqa.jpg';
import {useTranslation} from "react-i18next";

function References(){
    const {t, i18n} = useTranslation('common');

const drqa = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'DrQA',
    description:t('references.drqa.description'), 
    img:drqa_img,
    imageStart:'start',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://arxiv.org/pdf/1704.00051v2.pdf'
}


const wiki = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'MediaWiki API',
    description:t('references.wiki.description'), 
    img:wiki_img,
    imageStart:'row',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://www.mediawiki.org/wiki/API:Main_page'
}

const TFIDF = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'TF-IDF',
    description: t('references.tfidf.description'),
    img:tfidf_img,
    imageStart:'start',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://www.researchgate.net/publication/326425709_Text_Mining_Use_of_TF-IDF_to_Examine_the_Relevance_of_Words_to_Documents'
}

const BM25 = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'BM25',
    description: t('references.bm25.description'),
    img:bm25_img,
    imageStart:'row',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_921'
}

const word2vec = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'Word2vec',
    description: t('references.word2vec.description'),
    img:w2v_img,
    imageStart:'start',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://arxiv.org/abs/1301.3781'
}

const st = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'Sentence Transformers',
    description: t('references.st.description'),
    img:st_img,
    imageStart:'row',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://arxiv.org/abs/1908.10084'
}

const bert = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'BERT',
    description: t('references.bert.description'),
    img:bert_img,
    imageStart:'start',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://arxiv.org/abs/1908.10084'
}


const squad = {
    lightBg: true,
    lightText: false,
    lightTextDesc: false,
    buttonType:'btn--dark',
    topLine: '',
    headline: 'SQuAD',
    description: t('references.squad.description'),
    img:squad_img,
    imageStart:'row',
    alt:'/images/query-img.jpg',
    buttonLabel:'Source',
    buttonLink:'https://rajpurkar.github.io/SQuAD-explorer/'
}


    return (
        <div>
            <ReferenceObject {...drqa}></ReferenceObject>
            <ReferenceObject {...wiki}></ReferenceObject>
            <ReferenceObject {...TFIDF}></ReferenceObject>
            <ReferenceObject {...BM25}></ReferenceObject>
            <ReferenceObject {...word2vec}></ReferenceObject>
            <ReferenceObject {...st}></ReferenceObject>
            <ReferenceObject {...bert}></ReferenceObject>
            <ReferenceObject {...squad}></ReferenceObject>
        </div>

    )
        
    
}

export default References