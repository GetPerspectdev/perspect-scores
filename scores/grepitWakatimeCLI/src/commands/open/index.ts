import {Args, Command} from '@oclif/core'
import {spawnSync, execSync} from 'node:child_process'

export default class Test extends Command {
  static description = 'Open GrepIt local server in browser'

  static args = {
    path: Args.string(),
  }

  static examples = [`
  $ oex test
  `]

  async run(): Promise<void> {
    const ps = execSync('ps -ax | grep ../../proxy/server.js')
    const psMatches = ps.toString().match(/(\S+)/) || []
    const pid = psMatches[0]
    const lsof = execSync(`lsof -i -P -n | grep ${pid} | grep LISTEN`)
    const lsofMatches = lsof.toString().match(/\*:(\d+)/)
    const port = lsofMatches![1] || 8888
    spawnSync('open', [`http://localhost:${port}/`], {stdio: 'inherit'})
  }
}
