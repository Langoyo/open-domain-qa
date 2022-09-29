import React from 'react'
import './Button.css'
import { Link} from 'react-router-dom';
// Defining differnet style of buttons
const STYLES = ['btn--primary', 'btn--outline', 'btn--dark']
// Defining different sizes of buttons
const SIZES = ['btn--medium', 'btn--large']

const Button = ({
    // We export several values
        children, 
        type, 
        onClick, 
        buttonStyle, 
        buttonSize,
        link}
    
    
    ) => {
        // And use them to create the code of the button
        // Depending on what is passed on style or sizes are applied, with default values if nothing is passed 
        const checkButtonStyle = STYLES.includes(buttonStyle) ? buttonStyle : STYLES[0]
        const checkButtonSize = SIZES.includes(buttonSize) ? buttonSize: SIZES[0]
    
        return(    
        <Link to= {link} className='btn-mobile'>
        <button className={ checkButtonStyle + ' '+ checkButtonSize}  onClick={onClick}
            type = {type}>
            {children}
            
        </button>
        </Link>

    
        )
    }

export default Button 