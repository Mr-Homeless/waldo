document.addEventListener('DOMContentLoaded', () => {
  const interactiveElements = document.querySelectorAll('.card, .panel, .interactive, tr');
  
  interactiveElements.forEach(element => {
    // Track if mouse is over element
    let isHovering = false;
    
    element.addEventListener('mouseenter', () => {
      isHovering = true;
    });
    
    element.addEventListener('mouseleave', () => {
      isHovering = false;
    });
    
    element.addEventListener('mousemove', (e) => {
      const rect = element.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      element.style.setProperty('--mouse-x', `${x}px`);
      element.style.setProperty('--mouse-y', `${y}px`);
    });
    
    // Initialize mouse position to center
    element.style.setProperty('--mouse-x', '50%');
    element.style.setProperty('--mouse-y', '50%');
  });
});