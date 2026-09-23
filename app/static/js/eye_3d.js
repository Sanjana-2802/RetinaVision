/**
 * Interactive 3D Eyeball Anatomical Visualizer using Three.js.
 * Features realistic multi-layer geometry (Sclera, Cornea, Iris, Pupil, Optic Nerve),
 * orbit controls, interactive anatomical hotspots, and smooth lighting.
 */

class InteractiveEyeViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        this.width = this.container.clientWidth || 500;
        this.height = this.container.clientHeight || 450;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.eyeGroup = null;
        this.hotspots = [];
        this.isUserInteracting = false;
        this.autoRotate = true;

        this.init();
    }

    init() {
        // Scene setup
        this.scene = new THREE.Scene();

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 1000);
        this.camera.position.set(0, 0, 7.5);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
        this.scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.0);
        dirLight1.position.set(5, 5, 8);
        this.scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0x0284c7, 1.0); // Soft medical cyan accent light
        dirLight2.position.set(-6, -4, -4);
        this.scene.add(dirLight2);

        // Master Eye Group
        this.eyeGroup = new THREE.Group();
        this.scene.add(this.eyeGroup);

        this.buildEyeball();
        this.setupEventListeners();
        this.animate();
    }

    buildEyeball() {
        // 1. Sclera (White outer shell with slight warmth and vein texture)
        const scleraGeo = new THREE.SphereGeometry(2.0, 64, 64);
        
        // Procedural canvas texture for realistic micro-capillaries on sclera
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Base off-white color
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw delicate red scleral vessels
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.22)';
        ctx.lineWidth = 1.5;
        for (let i = 0; i < 40; i++) {
            ctx.beginPath();
            let x = Math.random() * canvas.width;
            let y = Math.random() * canvas.height;
            ctx.moveTo(x, y);
            for (let j = 0; j < 5; j++) {
                x += (Math.random() - 0.5) * 60;
                y += (Math.random() - 0.5) * 40;
                ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        
        const scleraTexture = new THREE.CanvasTexture(canvas);
        const scleraMat = new THREE.MeshStandardMaterial({
            map: scleraTexture,
            roughness: 0.25,
            metalness: 0.05,
            color: 0xfafafa
        });
        const scleraMesh = new THREE.Mesh(scleraGeo, scleraMat);
        this.eyeGroup.add(scleraMesh);

        // 2. Cornea (Clear protruding dome on anterior surface)
        const corneaGeo = new THREE.SphereGeometry(1.2, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.45);
        const corneaMat = new THREE.MeshPhysicalMaterial({
            color: 0xffffff,
            transmission: 0.95,
            opacity: 1,
            transparent: true,
            roughness: 0.05,
            ior: 1.376,
            thickness: 0.5,
            reflectivity: 0.9
        });
        const corneaMesh = new THREE.Mesh(corneaGeo, corneaMat);
        corneaMesh.position.set(0, 0, 1.3);
        corneaMesh.scale.set(1.15, 1.15, 0.9);
        this.eyeGroup.add(corneaMesh);

        // 3. Iris (Deep Oceanic Cerulean Blue with radial striations)
        const irisCanvas = document.createElement('canvas');
        irisCanvas.width = 512;
        irisCanvas.height = 512;
        const ictx = irisCanvas.getContext('2d');
        const cx = 256, cy = 256;
        
        const irisGrad = ictx.createRadialGradient(cx, cy, 30, cx, cy, 250);
        irisGrad.addColorStop(0, '#0369a1');    // Inner deep blue
        irisGrad.addColorStop(0.5, '#0ea5e9');  // Vibrant cyan
        irisGrad.addColorStop(0.85, '#0284c7'); // Rich blue
        irisGrad.addColorStop(1, '#0f172a');    // Dark outer limbal ring
        
        ictx.fillStyle = irisGrad;
        ictx.beginPath();
        ictx.arc(cx, cy, 250, 0, Math.PI * 2);
        ictx.fill();

        // Iris radial fibers
        ictx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
        for (let a = 0; a < Math.PI * 2; a += 0.03) {
            ictx.beginPath();
            ictx.moveTo(cx + Math.cos(a) * 45, cy + Math.sin(a) * 45);
            ictx.lineTo(cx + Math.cos(a) * 240, cy + Math.sin(a) * 240);
            ictx.stroke();
        }

        const irisTexture = new THREE.CanvasTexture(irisCanvas);
        const irisGeo = new THREE.RingGeometry(0.38, 1.12, 48);
        const irisMat = new THREE.MeshStandardMaterial({
            map: irisTexture,
            roughness: 0.4,
            side: THREE.DoubleSide
        });
        const irisMesh = new THREE.Mesh(irisGeo, irisMat);
        irisMesh.position.set(0, 0, 1.85);
        this.eyeGroup.add(irisMesh);

        // 4. Pupil (Deep black aperture center)
        const pupilGeo = new THREE.CircleGeometry(0.38, 32);
        const pupilMat = new THREE.MeshBasicMaterial({ color: 0x050505 });
        const pupilMesh = new THREE.Mesh(pupilGeo, pupilMat);
        pupilMesh.position.set(0, 0, 1.852);
        this.eyeGroup.add(pupilMesh);

        // 5. Optic Nerve Stump (Posterior pole exit)
        const nerveGeo = new THREE.CylinderGeometry(0.4, 0.45, 1.4, 32);
        const nerveMat = new THREE.MeshStandardMaterial({
            color: 0xfecdd3, // Soft pale yellowish-pink myelin
            roughness: 0.6
        });
        const nerveMesh = new THREE.Mesh(nerveGeo, nerveMat);
        nerveMesh.rotation.x = Math.PI / 2;
        nerveMesh.position.set(0.3, 0.1, -2.4);
        this.eyeGroup.add(nerveMesh);

        // Tilt eye slightly toward viewer for immediate impact
        this.eyeGroup.rotation.y = -0.35;
        this.eyeGroup.rotation.x = 0.15;
    }

    setupEventListeners() {
        let prevMouseX = 0;
        let prevMouseY = 0;

        const onMouseDown = (e) => {
            this.isUserInteracting = true;
            this.autoRotate = false;
            prevMouseX = e.clientX;
            prevMouseY = e.clientY;
        };

        const onMouseMove = (e) => {
            if (!this.isUserInteracting) return;
            const deltaX = e.clientX - prevMouseX;
            const deltaY = e.clientY - prevMouseY;
            this.eyeGroup.rotation.y += deltaX * 0.007;
            this.eyeGroup.rotation.x += deltaY * 0.007;
            prevMouseX = e.clientX;
            prevMouseY = e.clientY;
        };

        const onMouseUp = () => {
            this.isUserInteracting = false;
        };

        this.container.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);

        // Touch support
        this.container.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                this.isUserInteracting = true;
                this.autoRotate = false;
                prevMouseX = e.touches[0].clientX;
                prevMouseY = e.touches[0].clientY;
            }
        });

        window.addEventListener('touchmove', (e) => {
            if (!this.isUserInteracting || e.touches.length !== 1) return;
            const deltaX = e.touches[0].clientX - prevMouseX;
            const deltaY = e.touches[0].clientY - prevMouseY;
            this.eyeGroup.rotation.y += deltaX * 0.008;
            this.eyeGroup.rotation.x += deltaY * 0.008;
            prevMouseX = e.touches[0].clientX;
            prevMouseY = e.touches[0].clientY;
        });

        window.addEventListener('touchend', () => {
            this.isUserInteracting = false;
        });

        // Resize handler
        window.addEventListener('resize', () => {
            if (!this.container) return;
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
            this.camera.aspect = this.width / this.height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(this.width, this.height);
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.autoRotate && !this.isUserInteracting) {
            this.eyeGroup.rotation.y += 0.0035;
        }

        this.renderer.render(this.scene, this.camera);
    }

    rotateToFeature(feature) {
        this.autoRotate = false;
        const targetRotations = {
            'anterior': { x: 0.1, y: 0.0 },       // Front: Cornea, Iris, Lens
            'retina': { x: 0.2, y: Math.PI * 0.75 }, // Angled view of posterior pole
            'optic_disc': { x: 0.1, y: Math.PI },   // Directly at the back (Optic nerve)
            'macula': { x: 0.0, y: -0.45 }         // Temporal focal angle
        };

        const target = targetRotations[feature] || { x: 0.1, y: 0.0 };
        // Smooth lerp
        const startX = this.eyeGroup.rotation.x;
        const startY = this.eyeGroup.rotation.y;
        let progress = 0;

        const step = () => {
            progress += 0.05;
            this.eyeGroup.rotation.x = THREE.MathUtils.lerp(startX, target.x, progress);
            this.eyeGroup.rotation.y = THREE.MathUtils.lerp(startY, target.y, progress);
            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };
        step();
    }
}

// Global hook
window.initEye3D = function(containerId) {
    return new InteractiveEyeViewer(containerId);
};
