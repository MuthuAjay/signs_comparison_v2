export const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
  transition: { duration: 0.3 }
};

export const scaleIn = {
  initial: { opacity: 0, scale: 0.8 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.8 },
  transition: { duration: 0.4 }
};

export const slideIn = (direction: 'left' | 'right' | 'up' | 'down' = 'up') => {
  const directions = {
    left: { x: -50, y: 0 },
    right: { x: 50, y: 0 },
    up: { x: 0, y: 50 },
    down: { x: 0, y: -50 }
  };
  
  return {
    initial: { opacity: 0, ...directions[direction] },
    animate: { opacity: 1, x: 0, y: 0 },
    exit: { opacity: 0, ...directions[direction] },
    transition: { duration: 0.5 }
  };
};

export const staggerChildren = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export const glowAnimation = {
  animate: {
    boxShadow: [
      '0 0 20px rgba(0, 245, 255, 0.3)',
      '0 0 40px rgba(0, 245, 255, 0.5)',
      '0 0 20px rgba(0, 245, 255, 0.3)'
    ],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'easeInOut'
    }
  }
};

export const bounceIn = {
  initial: { opacity: 0, scale: 0.3 },
  animate: {
    opacity: 1,
    scale: [0.3, 1.05, 0.95, 1],
    transition: {
      duration: 0.6,
      times: [0, 0.3, 0.7, 1],
      ease: 'easeInOut'
    }
  }
};

export const clusteringAnimation = {
  grouping: {
    scale: [1, 1.1, 1],
    rotate: [0, 5, 0],
    transition: { duration: 0.8, ease: 'easeInOut' }
  },
  separating: {
    scale: [1, 0.9, 1],
    rotate: [0, -5, 0],
    transition: { duration: 0.8, ease: 'easeInOut' }
  }
};

export const matrixCellAnimation = {
  initial: { opacity: 0, scale: 0 },
  animate: (similarity: number) => ({
    opacity: 1,
    scale: 1,
    backgroundColor: `hsl(${similarity * 120}, 70%, 50%)`,
    transition: {
      duration: 0.5,
      delay: Math.random() * 0.3
    }
  }),
  hover: {
    scale: 1.1,
    zIndex: 10,
    transition: { duration: 0.2 }
  }
};