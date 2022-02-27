import discord
from discord.ext import commands
import asyncio

client = commands.Bot(command_prefix = "!")

@client.event
async def on_ready():
    pritn("test on_ready")
@client.command()
async def dm(ctx):
    await ctx.author.send("Hello, I am not available at this time")