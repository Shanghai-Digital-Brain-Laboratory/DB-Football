1. （方法1）利用官方环境设置中的`write_full_episode_dumps=True`保存官方的`.dump`文件。
    用`load_from_official_trace()`函数读取，传入visualizer回放。
    ```python
    from tracer import MatchTracer
    from v.visualizer import Visualizer

    tracer=MatchTracer.load_from_official_trace("data/episode_done_20220426-213219251284.dump")
    # disable_RGB=True 不加载RGB图像
    visualizer=Visualizer(tracer,disable_RGB=True)
    visualizer.run()
    ```
2. （方法2）参照test_tracer.py在env中记录整场游戏的进行情况。如果env启用了render，会把渲染的每一帧RGB图像保存下来，因为没压成video，也没有resize，所以导致trace的体积庞大，但是回放中可以看RGB图像。
    ```python
    from tracer import MatchTracer
    class FooballEnv:
        def __init__(self):
            self._env=...

        def reset(self):
            # no_frame=True 一定不保存RGB图像
            self.tracer=MatchTracer(no_frame=True)
            self._observations=self._env.reset()
        
        def step(self,actions):
            self.tracer.update(self._observations,actions)
            self._observations,reward,done,info=self._env.step(actions)
            if done:
                self.tracer.update(self._observations)
                self.tracer.save(fn=...)
    ```

    用`load()`函数读取，传入visualizer回放。
    ```python
    from tracer import MatchTracer
    from v.visualizer import Visualizer

    tracer=MatchTracer.load("data/random_play_trace.pkl")
    # disable_RGB=True 不加载RGB图像
    visualizer=Visualizer(tracer,disable_RGB=True)
    visualizer.run()
    ```