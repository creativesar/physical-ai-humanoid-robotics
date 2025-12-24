import React, { useEffect, useRef } from 'react';

const HeroSection = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null); // To store animation frame ID

  // Check if we're client-side (SSR support)
  const isClient = typeof window !== 'undefined';

  // OPTIMIZED: Disable Three.js on mobile and tablets for better performance
  const isMobile = isClient && (window.innerWidth < 1024 || /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent));

  useEffect(() => {
    if (!isClient || !canvasRef.current || isMobile) return;

    let animationActive = true;
    let renderer: any = null;
    let scene: any = null;
    let camera: any = null;
    let core: any = null;
    let arcs: any[] = [];
    let rings: any[] = [];
    let shocks: any[] = [];
    let fire: any = null;
    let fireGeo: any = null;
    let pos: Float32Array | null = null;
    let vel: Float32Array | null = null;

    // Initialize the animation once libraries are loaded
    const initAnimation = () => {
      if (!canvasRef.current || !(window as any).THREE) {
        console.error('Three.js library not loaded or canvas not available');
        return;
      }

      const canvas = canvasRef.current;
      renderer = new (window as any).THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);

      scene = new (window as any).THREE.Scene();
      camera = new (window as any).THREE.PerspectiveCamera(
        65,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );
      camera.position.z = 12;

      // Central exploding plasma orb
      core = new (window as any).THREE.Mesh(
        new (window as any).THREE.IcosahedronGeometry(1.5, 2),
        new (window as any).THREE.MeshBasicMaterial({
          color: 0xff4400,
          emissive: 0xff5500,
          emissiveIntensity: 3,
          transparent: true
        })
      );
      scene.add(core);

      // Thunder / lightning arcs - OPTIMIZED: Reduced from 12 to 6 arcs
      arcs = [];
      for(let i = 0; i < 6; i++){
        const points = [];
        for(let j = 0; j < 12; j++) points.push(new (window as any).THREE.Vector3()); // Reduced from 20 to 12 points
        const geo = new (window as any).THREE.BufferGeometry().setFromPoints(points);
        const mat = new (window as any).THREE.LineBasicMaterial({
          color: 0x00ffff,
          transparent: true,
          opacity: 0,
          linewidth: 4
        });
        const line = new (window as any).THREE.Line(geo, mat);
        scene.add(line);
        arcs.push(line);
      }

      // Orbiting energy rings - OPTIMIZED: Reduced from 5 to 3 rings, lower segment count
      rings = [];
      for(let i = 1; i <= 3; i++){
        const ring = new (window as any).THREE.Mesh(
          new (window as any).THREE.RingGeometry(i*1.5, i*1.5 + 0.15, 32), // Reduced segments from 64 to 32
          new (window as any).THREE.MeshBasicMaterial({
            color: 0x00ffff,
            side: (window as any).THREE.DoubleSide,
            transparent: true,
            opacity: 0.6
          })
        );
        ring.rotation.x = Math.PI / 2 + Math.random()*0.5;
        scene.add(ring);
        rings.push(ring);
      }

      // Shockwave rings - OPTIMIZED: Reduced segments from 64 to 32
      shocks = [];
      for(let i = 0; i < 2; i++){ // Reduced from 3 to 2
        const s = new (window as any).THREE.Mesh(
          new (window as any).THREE.RingGeometry(0.5, 8, 32), // Reduced segments
          new (window as any).THREE.MeshBasicMaterial({
            color: 0xff0088,
            side: (window as any).THREE.DoubleSide,
            transparent: true,
            opacity: 0
          })
        );
        s.rotation.x = -Math.PI/2;
        scene.add(s);
        shocks.push(s);
      }

      // Fire particles (explosion) - OPTIMIZED: Reduced from 800 to 200
      const fireCount = 200;
      fireGeo = new (window as any).THREE.BufferGeometry();
      pos = new Float32Array(fireCount * 3);
      vel = new Float32Array(fireCount * 3);
      for(let i = 0; i < fireCount; i++){
        pos[i*3] = 0;
        pos[i*3+1] = 0;
        pos[i*3+2] = 0;
        vel[i*3] = (Math.random()-0.5)*0.8;
        vel[i*3+1] = Math.random()*1.5;
        vel[i*3+2] = (Math.random()-0.5)*0.8;
      }
      fireGeo.setAttribute('position', new (window as any).THREE.BufferAttribute(pos, 3));
      const fireMat = new (window as any).THREE.PointsMaterial({
        color: 0xff4400,
        size: 0.3,
        transparent: true,
        opacity: 0,
        blending: (window as any).THREE.AdditiveBlending
      });
      fire = new (window as any).THREE.Points(fireGeo, fireMat);
      scene.add(fire);

      // Function to create blast animation using GSAP approach manually
      let blastActive = false;
      let lastBlastTime = 0;
      const blastInterval = 5000; // 5 seconds between blasts

      const performBlast = () => {
        if (!animationActive || blastActive) return;

        blastActive = true;
        lastBlastTime = Date.now();

        // Store original scale for later reset
        const originalScale = { x: core.scale.x, y: core.scale.y, z: core.scale.z };

        // Reset core to small size and start animation
        core.scale.set(0.1, 0.1, 0.1);

        // Use custom animation for core
        let growProgress = 0;
        const startTime = Date.now();
        const duration = 800; // 0.8 seconds
        const growAnimation = () => {
          if (!animationActive) return;

          const elapsed = Date.now() - startTime;
          growProgress = Math.min(elapsed / duration, 1);

          // Ease out function for growth
          const easeOut = 1 - Math.pow(1 - growProgress, 4);
          const scaleValue = 0.1 + easeOut * (2.8 - 0.1);
          core.scale.set(scaleValue, scaleValue, scaleValue);

          // Adjust emissive intensity
          const emissiveIntensity = 2 + (growProgress * 8); // From 2 to 10
          core.material.emissiveIntensity = emissiveIntensity;

          if (growProgress < 1) {
            requestAnimationFrame(growAnimation);
          }
        };
        requestAnimationFrame(growAnimation);

        // Thunder arcs explode outward after a delay
        setTimeout(() => {
          if (!animationActive) return;
          arcs.forEach(arc => {
            arc.material.opacity = 1;
          });

          // Fade out arcs
          setTimeout(() => {
            if (!animationActive) return;
            arcs.forEach(arc => {
              arc.material.opacity = 0;
            });
          }, 800);
        }, 300);

        // Orbiting rings spin fast
        const originalRotations = rings.map(ring => ring.rotation.z);
        let rotationProgress = 0;
        const rotationStartTime = Date.now();
        const rotationDuration = 2000; // 2 seconds
        const rotationAnimation = () => {
          if (!animationActive) return;

          const elapsed = Date.now() - rotationStartTime;
          rotationProgress = Math.min(elapsed / rotationDuration, 1);

          // Ease in-out for rotation
          const easeInOut = rotationProgress < 0.5
            ? 4 * rotationProgress * rotationProgress * rotationProgress
            : 1 - Math.pow(-2 * rotationProgress + 2, 3) / 2;

          const targetRotation = 12; // 12 radians total rotation
          rings.forEach((ring, i) => {
            ring.rotation.z = originalRotations[i] + easeInOut * targetRotation;
          });

          if (rotationProgress < 1) {
            requestAnimationFrame(rotationAnimation);
          }
        };
        requestAnimationFrame(rotationAnimation);

        // Fire particles blast
        setTimeout(() => {
          if (!animationActive) return;
          fire.material.opacity = 1;

          // Animate particles
          let particleProgress = 0;
          const particleStartTime = Date.now();
          const particleDuration = 2000; // 2 seconds
          const particleAnimation = () => {
            if (!animationActive) return;

            const elapsed = Date.now() - particleStartTime;
            particleProgress = Math.min(elapsed / particleDuration, 1);

            const easeOut = 1 - Math.pow(1 - particleProgress, 3);
            if (pos && vel && fireGeo) {
              const p = fireGeo.attributes.position.array;

              for(let i = 0; i < fireCount; i++){
                p[i*3]   += vel[i*3]   * 0.1 * easeOut;
                p[i*3+1] += vel[i*3+1] * 0.1 * easeOut;
                p[i*3+2] += vel[i*3+2] * 0.1 * easeOut;
              }

              fireGeo.attributes.position.needsUpdate = true;
            }

            if (particleProgress < 1) {
              requestAnimationFrame(particleAnimation);
            }
          };
          requestAnimationFrame(particleAnimation);

          // Fade out particles
          setTimeout(() => {
            if (!animationActive) return;
            let fadeProgress = 0;
            const fadeStartTime = Date.now();
            const fadeDuration = 1000; // 1 second
            const fadeAnimation = () => {
              if (!animationActive) return;

              const elapsed = Date.now() - fadeStartTime;
              fadeProgress = Math.min(elapsed / fadeDuration, 1);

              fire.material.opacity = 1 - fadeProgress;

              if (fadeProgress < 1) {
                requestAnimationFrame(fadeAnimation);
              }
            };
            requestAnimationFrame(fadeAnimation);
          }, 1500);
        }, 400);

        // Shockwaves
        setTimeout(() => {
          if (!animationActive) return;
          shocks[0].scale.set(1, 1, 1);
          shocks[0].material.opacity = 0.8;

          let shockProgress = 0;
          const shockStartTime = Date.now();
          const shockDuration = 2000; // 2 seconds
          const shockAnimation = () => {
            if (!animationActive) return;

            const elapsed = Date.now() - shockStartTime;
            shockProgress = Math.min(elapsed / shockDuration, 1);

            const scale = 1 + shockProgress * 19; // Scale from 1 to 20
            shocks[0].scale.set(scale, scale, scale);

            if (shockProgress < 1) {
              requestAnimationFrame(shockAnimation);
            }
          };
          requestAnimationFrame(shockAnimation);

          // Fade out shockwave
          setTimeout(() => {
            if (!animationActive) return;
            let fadeProgress = 0;
            const fadeStartTime = Date.now();
            const fadeDuration = 1500; // 1.5 seconds
            const fadeAnimation = () => {
              if (!animationActive) return;

              const elapsed = Date.now() - fadeStartTime;
              fadeProgress = Math.min(elapsed / fadeDuration, 1);

              shocks[0].material.opacity = 0.8 * (1 - fadeProgress);

              if (fadeProgress < 1) {
                requestAnimationFrame(fadeAnimation);
              }
            };
            requestAnimationFrame(fadeAnimation);
          }, 200);
        }, 600);

        setTimeout(() => {
          if (!animationActive) return;
          shocks[1].scale.set(1, 1, 1);
          shocks[1].material.opacity = 0.6;

          let shockProgress = 0;
          const shockStartTime = Date.now();
          const shockDuration = 2200; // 2.2 seconds
          const shockAnimation = () => {
            if (!animationActive) return;

            const elapsed = Date.now() - shockStartTime;
            shockProgress = Math.min(elapsed / shockDuration, 1);

            const scale = 1 + shockProgress * 24; // Scale from 1 to 25
            shocks[1].scale.set(scale, scale, scale);

            if (shockProgress < 1) {
              requestAnimationFrame(shockAnimation);
            }
          };
          requestAnimationFrame(shockAnimation);

          // Fade out shockwave
          setTimeout(() => {
            if (!animationActive) return;
            let fadeProgress = 0;
            const fadeStartTime = Date.now();
            const fadeDuration = 1500; // 1.5 seconds
            const fadeAnimation = () => {
              if (!animationActive) return;

              const elapsed = Date.now() - fadeStartTime;
              fadeProgress = Math.min(elapsed / fadeDuration, 1);

              shocks[1].material.opacity = 0.6 * (1 - fadeProgress);

              if (fadeProgress < 1) {
                requestAnimationFrame(fadeAnimation);
              }
            };
            requestAnimationFrame(fadeAnimation);
          }, 300);
        }, 900);

        setTimeout(() => {
          if (!animationActive) return;
          shocks[2].scale.set(1, 1, 1);
          shocks[2].material.opacity = 0.5;

          let shockProgress = 0;
          const shockStartTime = Date.now();
          const shockDuration = 2500; // 2.5 seconds
          const shockAnimation = () => {
            if (!animationActive) return;

            const elapsed = Date.now() - shockStartTime;
            shockProgress = Math.min(elapsed / shockDuration, 1);

            const scale = 1 + shockProgress * 29; // Scale from 1 to 30
            shocks[2].scale.set(scale, scale, scale);

            if (shockProgress < 1) {
              requestAnimationFrame(shockAnimation);
            }
          };
          requestAnimationFrame(shockAnimation);

          // Fade out shockwave
          setTimeout(() => {
            if (!animationActive) return;
            let fadeProgress = 0;
            const fadeStartTime = Date.now();
            const fadeDuration = 1800; // 1.8 seconds
            const fadeAnimation = () => {
              if (!animationActive) return;

              const elapsed = Date.now() - fadeStartTime;
              fadeProgress = Math.min(elapsed / fadeDuration, 1);

              shocks[2].material.opacity = 0.5 * (1 - fadeProgress);

              if (fadeProgress < 1) {
                requestAnimationFrame(fadeAnimation);
              }
            };
            requestAnimationFrame(fadeAnimation);
          }, 400);
        }, 1200);

        // Reset everything after animation completes
        setTimeout(() => {
          if (!animationActive) return;

          // Reset core scale
          if (originalScale) {
            core.scale.set(originalScale.x, originalScale.y, originalScale.z);
            core.material.emissiveIntensity = 3;
          }

          // Reset particles
          if (pos && fireGeo) {
            for(let i = 0; i < fireCount; i++){
              pos[i*3] = 0;
              pos[i*3+1] = 0;
              pos[i*3+2] = 0;
            }
            fireGeo.attributes.position.array = pos;
            fireGeo.attributes.position.needsUpdate = true;
            fire.material.opacity = 0;
          }

          // Hide arcs
          arcs.forEach(arc => {
            arc.material.opacity = 0;
          });

          blastActive = false;
        }, 3000);
      };

      // Start initial blast
      performBlast();

      // Set interval for repeating blasts
      const blastIntervalId = setInterval(() => {
        if (!animationActive || blastActive || Date.now() - lastBlastTime < blastInterval) {
          return;
        }
        performBlast();
      }, 100); // Check every 100ms

      // Animation loop
      const animate = () => {
        if (!animationActive) return;

        // Update lightning arcs every frame
        arcs.forEach((line, i) => {
          if (!line || !line.geometry || !line.geometry.attributes.position) return;
          const pos = line.geometry.attributes.position.array;
          const angle = i / arcs.length * Math.PI * 2;
          for(let j = 0; j < 20; j++){
            const t = j/19;
            const radius = t * 8 + Math.sin(Date.now()*0.005 + i)*2;
            pos[j*3]   = Math.sin(angle + t*5) * radius;
            pos[j*3+1] = (t-0.5)*6;
            pos[j*3+2] = Math.cos(angle + t*5) * radius;
          }
          line.geometry.attributes.position.needsUpdate = true;
        });

        // Slowly rotate rings
        rings.forEach(ring => {
          ring.rotation.z += 0.005;
        });

        if (renderer && scene && camera) {
          renderer.render(scene, camera);
        }

        animationRef.current = requestAnimationFrame(animate);
      };
      animate();

      // Handle window resize
      const handleResize = () => {
        if (!animationActive || !camera || !renderer) return;
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      };

      window.addEventListener('resize', handleResize);

      // Cleanup function
      return () => {
        animationActive = false;

        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }

        window.removeEventListener('resize', handleResize);
        clearInterval(blastIntervalId);

        // Dispose of Three.js objects
        if (renderer) {
          renderer.dispose();
        }
        if (scene) {
          // Dispose of geometries and materials
          scene.traverse((object: any) => {
            if (object.isMesh) {
              object.geometry.dispose();
              if (object.material) {
                object.material.dispose();
              }
            }
          });
        }
      };
    };

    // Initialize the animation directly since Three.js is loaded globally
    initAnimation();

    // Cleanup on unmount
    return () => {
      animationActive = false;

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isClient]);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Canvas for the 3D animation */}
      <canvas
        ref={canvasRef}
        id="c"
        className="absolute top-0 left-0 w-full h-full opacity-30"
      />

      {/* Luxury overlay gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black/40 to-black/80"></div>

      {/* Center content overlay */}
      <div className="absolute inset-0 flex items-center justify-center z-10 px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-6xl mx-auto">
          {/* Main heading with luxury styling */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold mb-6 leading-tight">
            <span className="bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent font-extrabold tracking-tight">
              Physical AI &
            </span>
            <br />
            <span className="bg-gradient-to-r from-amber-300 via-yellow-400 to-orange-500 bg-clip-text text-transparent font-extrabold tracking-tight relative">
              Humanoid Robotics
              <span className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-1/3 h-0.5 bg-gradient-to-r from-transparent via-amber-400 to-transparent"></span>
            </span>
          </h1>

          {/* Subtitle with premium styling */}
          <p className="text-base sm:text-lg md:text-xl lg:text-2xl text-gray-300 font-light mb-8 max-w-4xl mx-auto leading-relaxed tracking-wide">
            The definitive textbook for understanding the intersection of artificial intelligence and robotics,
            focusing on embodied intelligence and humanoid systems.
          </p>

          {/* Additional luxury subheading */}
          <div className="mt-12">
            <h2 className="text-lg sm:text-xl md:text-2xl text-gray-400 font-light tracking-widest uppercase mb-4">
              Advanced Intelligence Meets Physical Form
            </h2>
            <div className="flex items-center justify-center space-x-8 text-gray-500 text-sm">
              <span className="flex items-center">
                <span className="w-2 h-2 bg-amber-400 rounded-full mr-2"></span>
                ROS 2 Framework
              </span>
              <span className="flex items-center">
                <span className="w-2 h-2 bg-amber-400 rounded-full mr-2"></span>
                NVIDIA Isaac™ Platform
              </span>
              <span className="flex items-center">
                <span className="w-2 h-2 bg-amber-400 rounded-full mr-2"></span>
                Vision-Language-Action Systems
              </span>
            </div>
          </div>

          {/* CTA Button */}
          <div className="mt-16">
            <button className="bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white font-semibold py-4 px-10 rounded-full text-lg transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-amber-500/25 border border-amber-500/30">
              Explore the Future
            </button>
          </div>
        </div>
      </div>

      {/* Luxury corner accents */}
      <div className="absolute top-8 left-8 w-16 h-16 border-l-2 border-t-2 border-amber-400/30"></div>
      <div className="absolute top-8 right-8 w-16 h-16 border-r-2 border-t-2 border-amber-400/30"></div>
      <div className="absolute bottom-8 left-8 w-16 h-16 border-l-2 border-b-2 border-amber-400/30"></div>
      <div className="absolute bottom-8 right-8 w-16 h-16 border-r-2 border-b-2 border-amber-400/30"></div>
    </div>
  );
};

export default HeroSection;