import React from 'react'
import './Message.css'
import '../App.css'


const Message = ({
    // We export several values
        messageText,
        user    
    }
    
    ) => {    

        const userBubble = user === 'user' ? 'user-bubble' : 'elise-bubble'
        const userText = user === 'user' ? 'user-text' : 'elise-text'

        return(    
            <div className='message-line'>
                <div className={userBubble}>
                    <div className={userText}>{messageText}</div>
                </div>
           </div>
        )
    }

   

export default Message