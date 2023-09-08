grepit
=================

Nerd Status CLI

<!-- toc -->
* [Usage](#usage)
* [Commands](#commands)
<!-- tocstop -->
# Usage
<!-- usage -->
```sh-session
$ npm install -g grepit
$ grepit COMMAND
running command...
$ grepit (--version)
grepit/0.0.0 darwin-arm64 node-v16.15.0
$ grepit --help [COMMAND]
USAGE
  $ grepit COMMAND
...
```
<!-- usagestop -->
# Commands
<!-- commands -->
* [`grepit help [COMMANDS]`](#grepit-help-commands)
* [`grepit plugins`](#grepit-plugins)
* [`grepit plugins:install PLUGIN...`](#grepit-pluginsinstall-plugin)
* [`grepit plugins:inspect PLUGIN...`](#grepit-pluginsinspect-plugin)
* [`grepit plugins:install PLUGIN...`](#grepit-pluginsinstall-plugin-1)
* [`grepit plugins:link PLUGIN`](#grepit-pluginslink-plugin)
* [`grepit plugins:uninstall PLUGIN...`](#grepit-pluginsuninstall-plugin)
* [`grepit plugins:uninstall PLUGIN...`](#grepit-pluginsuninstall-plugin-1)
* [`grepit plugins:uninstall PLUGIN...`](#grepit-pluginsuninstall-plugin-2)
* [`grepit plugins:update`](#grepit-pluginsupdate)
* [`grepit start`](#grepit-start)
* [`grepit stop`](#grepit-stop)
* [`grepit test [PATH]`](#grepit-test-path)

## `grepit help [COMMANDS]`

Display help for grepit.

```
USAGE
  $ grepit help [COMMANDS] [-n]

ARGUMENTS
  COMMANDS  Command to show help for.

FLAGS
  -n, --nested-commands  Include all nested commands in the output.

DESCRIPTION
  Display help for grepit.
```

_See code: [@oclif/plugin-help](https://github.com/oclif/plugin-help/blob/v5.2.9/src/commands/help.ts)_

## `grepit plugins`

List installed plugins.

```
USAGE
  $ grepit plugins [--core]

FLAGS
  --core  Show core plugins.

DESCRIPTION
  List installed plugins.

EXAMPLES
  $ grepit plugins
```

_See code: [@oclif/plugin-plugins](https://github.com/oclif/plugin-plugins/blob/v2.4.7/src/commands/plugins/index.ts)_

## `grepit plugins:install PLUGIN...`

Installs a plugin into the CLI.

```
USAGE
  $ grepit plugins:install PLUGIN...

ARGUMENTS
  PLUGIN  Plugin to install.

FLAGS
  -f, --force    Run yarn install with force flag.
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Installs a plugin into the CLI.
  Can be installed from npm or a git url.

  Installation of a user-installed plugin will override a core plugin.

  e.g. If you have a core plugin that has a 'hello' command, installing a user-installed plugin with a 'hello' command
  will override the core plugin implementation. This is useful if a user needs to update core plugin functionality in
  the CLI without the need to patch and update the whole CLI.


ALIASES
  $ grepit plugins:add

EXAMPLES
  $ grepit plugins:install myplugin 

  $ grepit plugins:install https://github.com/someuser/someplugin

  $ grepit plugins:install someuser/someplugin
```

## `grepit plugins:inspect PLUGIN...`

Displays installation properties of a plugin.

```
USAGE
  $ grepit plugins:inspect PLUGIN...

ARGUMENTS
  PLUGIN  [default: .] Plugin to inspect.

FLAGS
  -h, --help     Show CLI help.
  -v, --verbose

GLOBAL FLAGS
  --json  Format output as json.

DESCRIPTION
  Displays installation properties of a plugin.

EXAMPLES
  $ grepit plugins:inspect myplugin
```

_See code: [@oclif/plugin-plugins](https://github.com/oclif/plugin-plugins/blob/v2.4.7/src/commands/plugins/inspect.ts)_

## `grepit plugins:install PLUGIN...`

Installs a plugin into the CLI.

```
USAGE
  $ grepit plugins:install PLUGIN...

ARGUMENTS
  PLUGIN  Plugin to install.

FLAGS
  -f, --force    Run yarn install with force flag.
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Installs a plugin into the CLI.
  Can be installed from npm or a git url.

  Installation of a user-installed plugin will override a core plugin.

  e.g. If you have a core plugin that has a 'hello' command, installing a user-installed plugin with a 'hello' command
  will override the core plugin implementation. This is useful if a user needs to update core plugin functionality in
  the CLI without the need to patch and update the whole CLI.


ALIASES
  $ grepit plugins:add

EXAMPLES
  $ grepit plugins:install myplugin 

  $ grepit plugins:install https://github.com/someuser/someplugin

  $ grepit plugins:install someuser/someplugin
```

_See code: [@oclif/plugin-plugins](https://github.com/oclif/plugin-plugins/blob/v2.4.7/src/commands/plugins/install.ts)_

## `grepit plugins:link PLUGIN`

Links a plugin into the CLI for development.

```
USAGE
  $ grepit plugins:link PLUGIN

ARGUMENTS
  PATH  [default: .] path to plugin

FLAGS
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Links a plugin into the CLI for development.
  Installation of a linked plugin will override a user-installed or core plugin.

  e.g. If you have a user-installed or core plugin that has a 'hello' command, installing a linked plugin with a 'hello'
  command will override the user-installed or core plugin implementation. This is useful for development work.


EXAMPLES
  $ grepit plugins:link myplugin
```

_See code: [@oclif/plugin-plugins](https://github.com/oclif/plugin-plugins/blob/v2.4.7/src/commands/plugins/link.ts)_

## `grepit plugins:uninstall PLUGIN...`

Removes a plugin from the CLI.

```
USAGE
  $ grepit plugins:uninstall PLUGIN...

ARGUMENTS
  PLUGIN  plugin to uninstall

FLAGS
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Removes a plugin from the CLI.

ALIASES
  $ grepit plugins:unlink
  $ grepit plugins:remove
```

## `grepit plugins:uninstall PLUGIN...`

Removes a plugin from the CLI.

```
USAGE
  $ grepit plugins:uninstall PLUGIN...

ARGUMENTS
  PLUGIN  plugin to uninstall

FLAGS
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Removes a plugin from the CLI.

ALIASES
  $ grepit plugins:unlink
  $ grepit plugins:remove
```

_See code: [@oclif/plugin-plugins](https://github.com/oclif/plugin-plugins/blob/v2.4.7/src/commands/plugins/uninstall.ts)_

## `grepit plugins:uninstall PLUGIN...`

Removes a plugin from the CLI.

```
USAGE
  $ grepit plugins:uninstall PLUGIN...

ARGUMENTS
  PLUGIN  plugin to uninstall

FLAGS
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Removes a plugin from the CLI.

ALIASES
  $ grepit plugins:unlink
  $ grepit plugins:remove
```

## `grepit plugins:update`

Update installed plugins.

```
USAGE
  $ grepit plugins:update [-h] [-v]

FLAGS
  -h, --help     Show CLI help.
  -v, --verbose

DESCRIPTION
  Update installed plugins.
```

_See code: [@oclif/plugin-plugins](https://github.com/oclif/plugin-plugins/blob/v2.4.7/src/commands/plugins/update.ts)_

## `grepit start`

Testing stuff

```
USAGE
  $ grepit start

DESCRIPTION
  Testing stuff

EXAMPLES
      $ oex start
```

_See code: [dist/commands/start/index.ts](https://github.com/narfdre/grepit/blob/v0.0.0/dist/commands/start/index.ts)_

## `grepit stop`

Testing stuff

```
USAGE
  $ grepit stop

DESCRIPTION
  Testing stuff

EXAMPLES
      $ oex start username
```

_See code: [dist/commands/stop/index.ts](https://github.com/narfdre/grepit/blob/v0.0.0/dist/commands/stop/index.ts)_

## `grepit test [PATH]`

Testing stuff

```
USAGE
  $ grepit test [PATH]

DESCRIPTION
  Testing stuff

EXAMPLES
    $ oex test
```

_See code: [dist/commands/test/index.ts](https://github.com/narfdre/grepit/blob/v0.0.0/dist/commands/test/index.ts)_
<!-- commandsstop -->
