import React, {useState} from 'react'
import {Link} from 'react-router-dom'
import Button from './Button'
import './Navbar.css'
import usa from '../images/united-states.png'
import spain from '../images/spain.png'
import {useTranslation} from "react-i18next";

function Navbar() {
    const [click, setClick] = useState(false);

    const handleClick = () => setClick(!click);
    const closeMobileMenu = () => setClick(false);

    const {t, i18n} = useTranslation('common');
    const [lanButton, setLanButton] = useState(false);



    function changeLang(lan){
        i18n.changeLanguage(lan)
        setLanButton(!lanButton);
        
            
    }

  return (
        <nav className="navbar">
            <div className="navbar-container">
                <Link to="/" className="navbar-logo">
                    ELISE
                </Link>
                <div className='menu-icon' onClick={handleClick}>
                </div>
                <ul className={click ? 'nav-menu active ': 'nav-menu'}>

                    <li className='nav-item'>
                    <Link to="/" id='nav-item-logo' className={"nav-links"}>
                    ELISE
                    </Link>

                    </li>
                    <li className='nav-item hide'>
                        <Link to='/' className=' nav-links' onClick={closeMobileMenu}>
                        {t('navbar.home')}
                        </Link>

                    </li>

                    {/* <li className='nav-item'>
                        <Link to='/chat' className=' nav-links' onClick={closeMobileMenu}>
                            Chat
                        </Link>
                    </li> */}

                    <li className='nav-item'>
                        <Link to='/query' className=' nav-links' onClick={closeMobileMenu}>
                        {t('navbar.playground')}
                        </Link>
                    </li>

                    <li className='nav-item'>
                        <Link to='/references' className=' nav-links' onClick={closeMobileMenu}>
                        {t('navbar.references')}
                        </Link>
                    </li>
                    <li className='nav-item'>
                        {lanButton?
                        <img src={spain} alt={'lang-image'} onClick={()=>changeLang('en')} className = 'lan-image'/>:
                        <img src={usa} alt={'lang-image'} onClick={ ()=>changeLang('es') } className = 'lan-image'/>

                    }
                    </li>

                
                </ul>
            </div>
        </nav>
  )
}

export default Navbar