
let isRunning=false,statusInterval=null,sleepyTimer=0;
const sleepyThreshold=5;
const alarmSound=new Audio('/static/22.mp3');

function startCamera(){
    const videoStream=document.getElementById('videoStream');
    const videoOverlay=document.getElementById('videoOverlay');
    const startBtn=document.getElementById('startBtn');
    const stopBtn=document.getElementById('stopBtn');
    videoStream.src='/video_feed';
    videoOverlay.style.display='none';
    startBtn.disabled=true;
    stopBtn.disabled=false;
    isRunning=true;
    statusInterval=setInterval(updateStatus,500);
}

function stopCamera(){
    const videoStream=document.getElementById('videoStream');
    const videoOverlay=document.getElementById('videoOverlay');
    const startBtn=document.getElementById('startBtn');
    const stopBtn=document.getElementById('stopBtn');
    fetch('/stop_camera').then(res=>res.json()).then(data=>{
        videoStream.src='';
        videoOverlay.style.display='flex';
        startBtn.disabled=false;
        stopBtn.disabled=true;
        isRunning=false;
        if(statusInterval) clearInterval(statusInterval);
        resetUI();
        sleepyTimer=0;
    });
}

function updateStatus(){
    if(!isRunning) return;
    fetch('/status').then(res=>res.json()).then(data=>{
        updateStatusDisplay(data.status);
        updateEyeScores(data);
        updateRegions(data);
        checkSleepyAlert(data);
    }).catch(err=>console.error(err));
}

function checkSleepyAlert(data){
    const statusDisplay=document.getElementById('statusDisplay');
    if(data.status==='SLEEPY'){
        sleepyTimer+=0.5;
        if(sleepyTimer>=sleepyThreshold){
            if(!alarmSound.paused) return;
            alarmSound.play();
            statusDisplay.style.boxShadow='0 0 20px 5px red';
        }
    } else {
        sleepyTimer=0;
        statusDisplay.style.boxShadow='none';
        alarmSound.pause();
        alarmSound.currentTime=0;
    }
}

function updateStatusDisplay(status){
    const statusDisplay=document.getElementById('statusDisplay');
    const statusText=document.getElementById('statusText');
    const statusIcon=statusDisplay.querySelector('.status-icon');
    statusDisplay.className='status-display';
    if(status==='AWAKE'){
        statusDisplay.classList.add('awake');
        statusIcon.textContent='✓';
        statusText.innerHTML='Awake';
        statusDisplay.querySelector('.status-subtext').textContent='AWAKE';
    } else if(status==='SLEEPY'){
        statusDisplay.classList.add('sleepy');
        statusIcon.textContent='⚠';
        statusText.innerHTML='Sleepy';
        statusDisplay.querySelector('.status-subtext').textContent='SLEEPY';
    } else{
        statusDisplay.classList.add('no-face');
        statusIcon.textContent='⚪';
        statusText.innerHTML='No Face Detected';
        statusDisplay.querySelector('.status-subtext').textContent='No Face Detected';
    }
}

function updateEyeScores(data){
    const rightScore=(data.right_eye_score*100).toFixed(1);
    const leftScore=(data.left_eye_score*100).toFixed(1);
    const avgScore=(data.avg_score*100).toFixed(1);
    document.getElementById('rightEyeScore').textContent=data.status!=='No Face'?rightScore+'%':'--';
    document.getElementById('leftEyeScore').textContent=data.status!=='No Face'?leftScore+'%':'--';
    document.getElementById('avgScore').textContent=data.status!=='No Face'?avgScore+'%':'--';
    document.getElementById('rightEyeProgress').style.width=rightScore+'%';
    document.getElementById('leftEyeProgress').style.width=leftScore+'%';
    document.getElementById('avgProgress').style.width=avgScore+'%';
    const avgProgress=document.getElementById('avgProgress');
    avgProgress.style.background=data.avg_score>=0.5?'#10b981':'#ef4444';
}

function updateRegions(data){
    const regionsCard=document.getElementById('regionsCard');
    if(data.right_eye_region&&data.left_eye_region){
        regionsCard.style.display='block';
        const r=data.right_eye_region,l=data.left_eye_region;
        document.getElementById('rightRegion').innerHTML=`X: ${r.x}px<br>Y: ${r.y}px<br>W: ${r.width}px<br>H: ${r.height}px`;
        document.getElementById('leftRegion').innerHTML=`X: ${l.x}px<br>Y: ${l.y}px<br>W: ${l.width}px<br>H: ${l.height}px`;
    } else regionsCard.style.display='none';
}

function resetUI(){
    document.getElementById('statusText').innerHTML='No Face Detected';
    document.getElementById('rightEyeScore').textContent='--';
    document.getElementById('leftEyeScore').textContent='--';
    document.getElementById('avgScore').textContent='--';
    document.getElementById('rightEyeProgress').style.width='0%';
    document.getElementById('leftEyeProgress').style.width='0%';
    document.getElementById('avgProgress').style.width='0%';
    document.getElementById('regionsCard').style.display='none';
    document.getElementById('statusDisplay').style.boxShadow='none';
}
