import React from 'react';
import './Footer.css';
import { Link } from 'react-router-dom';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import InstagramIcon from '@mui/icons-material/Instagram';
import GitHubIcon from '@mui/icons-material/GitHub';
import TwitterIcon from '@mui/icons-material/Twitter';
import FavoriteIcon from '@mui/icons-material/Favorite';
import {useTranslation} from "react-i18next";


function Footer() {
  const {t, i18n} = useTranslation('common');

  return (
    <div className='footer-container'>
      <section className='footer-subscription'>
        <p className='footer-subscription-heading'>
        {t('footer.big')}
        </p>
        <p className='footer-subscription-text'>
        {t('footer.small1')}<FavoriteIcon></FavoriteIcon>{t('footer.small2')}
        </p>
        
      </section>
      {/* <div class='footer-links'>
        <div className='footer-link-wrapper'>
          <div class='footer-link-items'>
            <h2>About Us</h2>
            <Link to='/sign-up'>How it works</Link>
            <Link to='/'>Testimonials</Link>
            <Link to='/'>Careers</Link>
            <Link to='/'>Investors</Link>
            <Link to='/'>Terms of Service</Link>
          </div>
          <div class='footer-link-items'>
            <h2>Contact Us</h2>
            <Link to='/'>Contact</Link>
            <Link to='/'>Support</Link>
            <Link to='/'>Destinations</Link>
            <Link to='/'>Sponsorships</Link>
          </div>
        </div>
        <div className='footer-link-wrapper'>
          <div class='footer-link-items'>
            <h2>Videos</h2>
            <Link to='/'>Submit Video</Link>
            <Link to='/'>Ambassadors</Link>
            <Link to='/'>Agency</Link>
            <Link to='/'>Influencer</Link>
          </div>
          <div class='footer-link-items'>
            <h2>Social Media</h2>
            <Link to='/'>Instagram</Link>
            <Link to='/'>Facebook</Link>
            <Link to='/'>Youtube</Link>
            <Link to='/'>Twitter</Link>
          </div>
        </div>
      </div> */}
      <section class='social-media'>
        <div class='social-media-wrap'>
          {/* <div class='footer-logo'>
            <Link to='/' className='social-logo'>
              TRVL
              <i class='fab fa-typo3' />
            </Link>
          </div> */}
          <small class='website-rights'>ELISE - 2022</small>
          <div class='social-icons'>
            <a
              class='social-icon-link facebook'
              href='https://www.linkedin.com/in/langoyo/?originalSubdomain=es'
              target='_blank'
              aria-label='Facebook'
            >
              <LinkedInIcon/>
            </a>
            <a
              class='social-icon-link youtube'
              href='https://github.com/Langoyo'
              target='_blank'
              aria-label='Youtube'
            >
              <GitHubIcon/>
            </a>
            <a
              class='social-icon-link twitter'
              href='/'
              target='_blank'
              aria-label='Twitter'
            >
              <InstagramIcon/>
            </a>
            <a
              class='social-icon-link twitter'
              href='/'
              target='_blank'
              aria-label='LinkedIn'
            >
              <TwitterIcon/>
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Footer;