import React, { useEffect, useRef } from 'react';

const HeroSection = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null); // To store animation frame ID

  // Check if we're client-side (SSR support)
  const isClient = typeof window !== 'undefined';

  useEffect(() => {
    if (!isClient || !canvasRef.current) return;

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

      // Thunder / lightning arcs
      arcs = [];
      for(let i = 0; i < 12; i++){
        const points = [];
        for(let j = 0; j < 20; j++) points.push(new (window as any).THREE.Vector3());
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

      // Orbiting energy rings
      rings = [];
      for(let i = 1; i <= 5; i++){
        const ring = new (window as any).THREE.Mesh(
          new (window as any).THREE.RingGeometry(i*1.2, i*1.2 + 0.15, 64),
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

      // Shockwave rings
      shocks = [];
      for(let i = 0; i < 3; i++){
        const s = new (window as any).THREE.Mesh(
          new (window as any).THREE.RingGeometry(0.5, 8, 64),
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

      // Fire particles (explosion)
      const fireCount = 800;
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
    <div className="relative w-full h-screen overflow-hidden bg-black">
      {/* Canvas for the 3D animation */}
      <canvas
        ref={canvasRef}
        id="c"
        className="absolute top-0 left-0 w-full h-full"
      />

      {/* Center content overlay */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
        <div className="text-center px-4">
          <h1 className="text-5xl md:text-7xl lg:text-9xl font-black tracking-tighter bg-gradient-to-b from-orange-400 via-red-500 to-pink-600 bg-clip-text text-transparent animate-pulse mb-4">
            PLASMA BLAST
          </h1>
          <p className="text-xl md:text-2xl lg:text-4xl text-cyan-400 tracking-widest font-mono">
            T H U N D E R   C O R E
          </p>
          <div className="mt-8">
            <h2 className="text-2xl md:text-4xl font-bold text-white" style={{ fontFamily: 'Sora, sans-serif' }}>
              Physical AI & Humanoid Robotics
            </h2>
            <p className="text-lg md:text-xl text-gray-300 mt-4 max-w-3xl mx-auto" style={{ fontFamily: 'Inter, sans-serif' }}>
              The definitive textbook for understanding the intersection of artificial intelligence and robotics,
              focusing on embodied intelligence and humanoid systems.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;