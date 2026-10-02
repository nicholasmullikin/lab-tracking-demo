// Capture real Rerun frames for the README headline; no fabricated motion.
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.BATTLE_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const [url, output] = process.argv.slice(2);
  if (!url || !output) throw new Error('Usage: node capture_pipette_demo_gif.cjs URL OUTPUT');
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  try {
    await page.goto(url);
    await page.waitForFunction(()=>!document.getElementById('demo-masks').disabled,null,{timeout:240000});
    await page.waitForTimeout(1000);
    const metadata=await page.evaluate(()=>({recordingId:window._handle.get_active_recording_id(),
      range:window._handle.get_timeline_time_range(window._handle.get_active_recording_id(),'raw_frame')}));
    const frames=[];
    for (let index=0;index<50;index++) {
      const frame=index*6;
      await page.evaluate(frame=>{
        const h=window._handle,id=h.get_active_recording_id();
        h.set_playing(id,false);h.set_active_timeline(id,'raw_frame');h.set_time_for_timeline(id,'raw_frame',frame);
      },frame);
      await page.waitForTimeout(180);
      await page.screenshot({path:path.join(output,`frame-${String(index).padStart(3,'0')}.png`),
        clip:{x:0,y:0,width:1280,height:726}});
      frames.push(frame);
    }
    const report={url,...metadata,rawFrames:frames,nativeFps:30000/1001,gifFps:5,
      cameraImageHz:2,viewport:{width:1280,height:900},crop:{x:0,y:0,width:1280,height:726},errors};
    fs.writeFileSync(path.join(output,'capture.json'),JSON.stringify(report,null,2)+'\n');
    if(errors.length)throw new Error(errors.join('\n'));
    console.log(`Captured ${frames.length} real viewer frames from ${metadata.recordingId}`);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
