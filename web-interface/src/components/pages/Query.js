import React from 'react'
import { useState, useEffect } from "react";
import 'bootstrap/dist/css/bootstrap.min.css';
import './Query.css'
import Message from "../Message";
import Form from 'react-bootstrap/Form'
import Button from 'react-bootstrap/Button'
import { renderMatches } from "react-router-dom";
import ReactDOM from "react-dom";
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { useSpeechRecognition } from 'react-speech-kit'
import profilepic from '../../images/profile.png'
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import VoiceOverOffIcon from '@mui/icons-material/VoiceOverOff';
import { useSpeechSynthesis } from 'react-speech-kit';
import { useTranslation } from "react-i18next";


// const AnswerCustom = row => <div class="react-grid-multiline-content"> <p>{row.answer}</p></div>;
// const ConentCustom = row => <div class="react-grid-multiline-content"> <p>{row.content}</p></div>;
// const EmailCustom = row => <div class="react-grid-multiline-content"> <p>{row.url}</p></div>;




function Query() {
    const { t, i18n } = useTranslation('common');


    var data = [

    ]

    var messages = [
        // {
        //     messageText: t('query.message'),
        //     user: 'Elise'
        // },
    ]



    const [query, setQuery] = useState("");
    const [reader, setReader] = useState("BERT");
    const [ranker, setRanker] = useState("BM25");
    const [number, setNumber] = useState(10);


    const [result, setCurrentResult] = useState("");
    var results = ''
    const [showAdvanced, setAdvanced] = useState(false)

    const [myData, setMyData] = useState(data);
    const [myMessages, setMessages] = useState([]);
    const [voice, setVoice] = useState(false);
    const { speak, voices } = useSpeechSynthesis();

    const [statusLastResponse, setStatusLastResponse] = useState(true);
    function findVoice(voice) {
        if (i18n.language === 'en') {
            return voice.name.includes('ngl')
        } else {
            return voice.name.includes('spa')
        }

    }

    const getData = () => {
        // https://k8wpmesah3.execute-api.eu-west-1.amazonaws.com/dev/qa
        fetch("https://k8wpmesah3.execute-api.eu-west-1.amazonaws.com/dev/qa", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Access-Control-Request-Method': 'OPTIONS,POST,GET,ANY,DELETE',
                // "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token"
            },
            body: JSON.stringify({ 'query': query, 'reader': reader, 'ranker': ranker, 'number': number })


        }).then(function (response) {
            console.log(response.data)
                if (response.ok) {
                    console.log(statusLastResponse)
                    response = response.json()
                    console.log(response)
                    return response;
                    
                }
                throw new Error('Error fetching answer')
                

            }).then(function (myJson) {
                    results = myJson

                    console.log(myData)
                    messages = [...messages.slice(0, messages.length - 1)]
                    messages = messages.concat([{ messageText: results[0].answer, user: 'elise' }])
                    setMessages(messages)

                    if (voice) {

                        speak({ text: results[0].answer, voice: voices.find(findVoice) })
                        console.log(messages)
                    }
                    setMyData(results)


            }).catch(error => {
                console.log(error)
                messages = [...messages.slice(0, messages.length - 1)]

                var errorMessage = 'An error occurred. Please try again or ask another question.'
                if (i18n.language === 'es') {
                    errorMessage = 'Ocurrio un error. Prueba otra vez o introduce otra pregunta.'
                }

                messages = messages.concat([{ messageText: errorMessage, user: 'elise' }])

                if (voice) {
                    speak({ text: errorMessage, voice: voices.find(findVoice) })
                    console.log(messages)
                }
                setMessages(messages)
            });
    }
    async function handleSubmit(e) {
        e.preventDefault();
        console.log(query + ranker + reader + number)
        // joined = messages.concat([{ messageText: query, user: 'user' }])
        // setMessages(joined)

        if (query !== "") {
            messages = [...myMessages]
            messages = messages.concat([{ messageText: query, user: 'user' }])
            messages = messages.concat([{ messageText: '...', user: 'Elise' }])
            setMessages(messages)
            console.log(voices)
            setMyData([])

            if (voice) {

                // speak({text:query,voice: voices.find(findVoice)})          

            }

            getData()

        }



    }

    function renderMessages() {
        const items = myMessages.map((item) =>
            (<Message messageText={item.messageText} user={item.user}></Message>));
        return items;
    };

    function changeVisibility() {
        setAdvanced(!showAdvanced)


    }


    function changeVoice() {
        setVoice(!voice)
    }
    const [inputText, setInputText] = useState('');
    const { listen, listening, stop } = useSpeechRecognition({
        onResult: (result) => {
            setInputText(result);
            setQuery(result);
        },
    });
    function listenLan() {


        if (i18n.language === 'en') {
            return { label: 'English', value: 'en-AU' }
        } else {
            return { label: 'Español', value: 'es-MX' }
        }
    }





    return (

        <div className='query-warpper'>
            {/* <button onClick={handleSpeak}>speak</button>
            <textarea value={transcript}>{transcript}</textarea> */}


            <div className='chat-wrapper'>
                <div className='chat-header navbar-expand-lg navbar-light '>
                    <img src={profilepic} alt={'profile-pic'} className='profile-img header-item' />
                    <div className='name header-item'>Elise</div>
                    <div className='right-group float-right'>
                        <div className='header-item voice-image' onClick={changeVoice}>{voice ? <RecordVoiceOverIcon /> : <VoiceOverOffIcon />}</div>
                        {/* <div  className='header-item'></div> */}
                    </div>

                </div>

                <div className='chat-box'>
                    <Message messageText={t('query.message')} user={'Elise'}></Message>
                    {renderMessages()}
                </div>


                <div className='chat-input'>

                    <Form className='chat-form' onSubmit={e => handleSubmit(e)}>
                        <button className="send-button" type="button" onClick={listenLan()} onMouseDown={listen} onMouseUp={stop}> <MicIcon className="icon" /></button>
                        <Form.Control className="input-text"
                            placeholder={t('query.placeholder')}
                            type="text"
                            value={query}
                            onChange={e => setQuery(e.target.value)} />
                        <button className="send-button" type='submit'><SendIcon className='icon' /></button>
                    </Form>

                </div>
            </div>

            <div className='collapsible' onClick={changeVisibility}>
                {t('query.options')} {showAdvanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </div>
            {showAdvanced
                ? <div id='show-advanced'>
                    <div className='form-wrapper'>
                        {/* <div className='search-box' onChange={query}>
                        <label for="fname">Question</label>
                        <input onChange={e => setQuery(e.target.value)} type="text" id="fname" name="fname"></input>
    
                    </div> */}

                        {/* <button type="button" className="collapsible">More Options ^</button> */}
                        <div className="content">


                            <Form onSubmit={e => handleSubmit(e)}>
                                <Form.Group>
                                    <Form.Label>{t('query.form-input')}</Form.Label>
                                    <Form.Control type="text"
                                        className="input-advanced"
                                        onChange={e => setQuery(e.target.value)} />

                                </Form.Group>

                                <Form.Group className='radial-box'>
                                    <label for="fname">{t('query.ranker')}</label>
                                    <Form.Check label="TF-IDF" checked={ranker === 'TFIDF'} value="TFIDF" onClick={e => { setRanker(e.target.value) }} />
                                    <Form.Check label="BM25" value="BM25" checked={ranker === 'BM25'} onClick={e => { setRanker(e.target.value) }} />
                                    <Form.Check label="Sentence Transformers" disabled="disabled" value="ST" checked={ranker === 'ST'} onClick={e => { setRanker(e.target.value) }} />
                                    <Form.Check label="Word2vec embeddings" disabled="disabled" value="w2v" checked={ranker === 'w2v'} onClick={e => { setRanker(e.target.value) }} />
                                </Form.Group>
                                <Form.Group className='radial-box'>
                                    <label for="fname">{t('query.reader')}</label>
                                    <Form.Check label="BERT" checked={reader === 'BERT'} value="BERT" onClick={() => setReader('BERT')} />
                                    <Form.Check label="LSTM + attention" checked={reader === 'LSTM'} value="LSTM" onClick={() => setReader('LSTM')} />
                                </Form.Group>
                                <Form.Group className='radial-box'>
                                    <label for="fname">{t('query.n-results')}</label>
                                    <Form.Select className='form-select' aria-label="Default select example">
                                        <option value="1" onClick={() => setNumber('1')}>1</option>
                                        <option value="2" onClick={() => setNumber('2')}>2</option>
                                        <option value="3" onClick={() => setNumber('3')}>3</option>
                                        <option value="4" onClick={() => setNumber('4')}>4</option>
                                        <option value="5" onClick={() => setNumber('5')}>5</option>
                                        <option value="6" onClick={() => setNumber('6')}>6</option>
                                        <option value="7" onClick={() => setNumber('7')}>7</option>
                                        <option value="8" onClick={() => setNumber('8')}>8</option>
                                        <option value="9" onClick={() => setNumber('9')}>9</option>
                                        <option value="10" onClick={() => setNumber('10')}>10</option>
                                    </Form.Select>
                                </Form.Group>
                                <Button type='submit' className={'button'} variant="outline-light" >{t('query.button-submit')}</Button>{' '}
                            </Form>
                        </div>
                    </div>
                    <div className='table-wrapper table-responsive'>
                        {/* score, title, content, url, answer */}
                        {/* <DataTable
                        responsive = {true}
                        highlightOnHover={false}
                        columns={my_columns}
                        data={myData}
                        customStyles={customStyles} /> */}
                        <table className='table table-hover'>
                            <thead>
                                <tr>
                                    <th scope='col'>{'#'}</th>
                                    <th scope='col'>{t('query.header-answer')}</th>
                                    <th scope='col'>{t('query.header-title')}</th>
                                    <th scope='col'>{t('query.header-passage')}</th>
                                    <th scope='col'>{t('query.header-link')}</th>
                                    <th scope='col'>{t('query.header-score')}</th>
                                </tr>
                            </thead>
                            <tbody>

                                {myData.map((item, index) => {
                                    return (
                                        <tr className="remove-row" key={item.score}>
                                            <td>{index + 1}</td>
                                            <td>{item.answer}</td>
                                            <td>{item.title}</td>
                                            <td>{item.content}</td>
                                            <td><a href={item.url} target="_blank">{item.url}</a></td>
                                            <td>{item.score}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>

                    </div>
                </div>
                : null}




        </div>

    )
}

export default Query