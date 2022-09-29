import React from 'react'
import './ChatBox.css'
import '../App.css'
import Button from './Button'

function ChatBox() {
  return (
    <div className = 'hero-container'>
        {/* <video src="/videos/video-2.mp4" autoPlay loop muted/> */}
        <h2>ELISE</h2>
        <p>An Open Domain Question Answering System</p>
            <div className='hero-btns'>
                 <Button classnmae='bins' buttonStyle='btn--outline' buttonSize='btn--large'>
                        ASK SOMETHING
                 </Button>
                 <Button classnmae='bins' buttonStyle='btn--primary' buttonSize='btn--large'>
                        HOW IT WORKS? 
                 </Button>
            </div> 
    </div>
  )
}

export default ChatBox