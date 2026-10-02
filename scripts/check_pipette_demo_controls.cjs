// Exercise the public demo controls in a real browser.
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.BATTLE_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const [url, output] = process.argv.slice(2);
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({executablePath:'/usr/bin/google-chrome', headless:true,
    args:['--no-sandbox', ...(process.env.BATTLE_PUBLIC_IP ?
      [`--host-resolver-rules=MAP ${new URL(url).hostname} ${process.env.BATTLE_PUBLIC_IP}`] : [])]});
  const page = await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(url);
    await page.waitForFunction(() => !document.getElementById('demo-masks').disabled, null, {timeout:240000});
    await page.evaluate(() => {
      const h=window._handle,id=h.get_active_recording_id();
      h.set_playing(id,false);h.set_active_timeline(id,'raw_frame');h.set_time_for_timeline(id,'raw_frame',100);
    });
    await page.waitForTimeout(1000);
    await page.screenshot({path:path.join(output,'orientation.png')});
    const checks=[];
    async function capture(name) {
      await page.waitForTimeout(1000);
      const state=await page.evaluate(() => {
        const h=window._handle,id=h.get_active_recording_id();
        return {recordingId:id,frame:h.get_time_for_timeline(id,'raw_frame'),playing:h.get_playing(id),
          timeline:h.get_active_timeline(id),view:document.getElementById('demo-view').value,masks:document.getElementById('demo-masks').checked};
      });
      if(state.frame!==100 || state.playing || state.timeline !== 'raw_frame') throw new Error('Control changed frame or playback: '+JSON.stringify(state));
      checks.push({name,...state});
      await page.screenshot({path:path.join(output,name+'.png')});
    }
    await page.selectOption('#demo-view','cameras');
    await capture('cameras-masks-on');
    await page.uncheck('#demo-masks');
    await capture('cameras-masks-off');
    await page.check('#demo-masks');
    await capture('cameras-masks-restored');
    await page.selectOption('#demo-view','orientation');
    await page.uncheck('#demo-masks');
    await capture('orientation-masks-off');
    await page.selectOption('#demo-view','evidence');
    await capture('evidence');
    await page.check('#demo-masks');
    await page.selectOption('#demo-view','cameras');
    await capture('cameras-masks-restored-again');
    const report={url, publicRelayIp:process.env.BATTLE_PUBLIC_IP||null,checks,errors,
      note:'Screenshots must be inspected to confirm visibility and restored masks.'};
    fs.writeFileSync(path.join(output,'controls-check.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
    if(errors.length) throw new Error('JavaScript errors');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
